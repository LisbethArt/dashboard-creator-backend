import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from supabase import Client

from app.config import Settings, get_settings
from app.deps import get_supabase
from app.schemas import ALLOWED_EXTENSIONS, AnalyzeResponse
from app.services.llm import generate_chart_suggestions, generate_column_short_labels
from app.services.profile import (
    build_dataframe_client_dataset,
    build_dataframe_profile,
    parse_tabular_bytes,
)
from app.services.storage import (
    new_upload_id,
    persist_upload_bundle,
    storage_object_paths,
)

router = APIRouter(tags=["analyze"])
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_upload(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    client: Client = Depends(get_supabase),
) -> AnalyzeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = file.filename.lower().rsplit(".", 1)
    if len(suffix) != 2:
        raise HTTPException(status_code=400, detail="Filename must include an extension")
    ext = f".{suffix[1]}"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx uploads are supported")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload")

    media = file.content_type or ("text/csv" if ext == ".csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    try:
        frame = parse_tabular_bytes(raw, ext)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    profile = build_dataframe_profile(frame)
    try:
        suggestions = generate_chart_suggestions(profile, settings)
    except RuntimeError as exc:
        msg = str(exc)
        lowered = msg.lower()
        if (
            "429" in msg
            or "resource_exhausted" in lowered
            or "quota exceeded" in lowered
            or ("quota" in lowered and "gemini" in lowered)
            or "rate limit" in lowered
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Cuota o límite de la API de Gemini alcanzado en todos los modelos probados "
                    "automáticamente en el servidor (económicos primero y, en última instancia, "
                    "modelos más potentes). Espere uno o dos minutos o revise su plan en "
                    "Google AI Studio (https://aistudio.google.com)."
                ),
            ) from exc
        raise HTTPException(status_code=502, detail=f"LLM suggestion failure: {exc}") from exc

    upload_id = new_upload_id()
    original_key, parquet_key = storage_object_paths(upload_id, ext)
    label_hints = generate_column_short_labels(profile, list(frame.columns)[:40], settings)
    logger.debug("Merged %s IA short column labels", len(label_hints))
    client_dataset = build_dataframe_client_dataset(frame, display_label_hints=label_hints)
    try:
        persist_upload_bundle(
            client,
            settings.supabase_bucket,
            upload_id,
            original_key,
            parquet_key,
            frame,
            raw,
            media,
            suggestions,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase persistence failure: {exc}") from exc

    return AnalyzeResponse(upload_id=upload_id, suggestions=suggestions, dataset=client_dataset)
