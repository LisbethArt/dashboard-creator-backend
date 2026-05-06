import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
import pandas as pd
from supabase import Client

from app.config import Settings, get_settings
from app.deps import get_supabase
from app.schemas import ALLOWED_EXTENSIONS, AnalyzeResponse, ChartSuggestion
from app.services.aggregation import aggregate_chart_dataframe
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


def _raise_user_friendly_llm_error(exc: RuntimeError) -> None:
    msg = str(exc)
    lowered = msg.lower()
    quota_or_rate = (
        "429" in msg
        or "resource_exhausted" in lowered
        or "resource exhausted" in lowered
        or "quota exceeded" in lowered
        or ("quota" in lowered and "gemini" in lowered)
        or "rate limit" in lowered
        or "too many requests" in lowered
    )
    if quota_or_rate:
        raise HTTPException(
            status_code=429,
            detail=(
                "El modelo de IA está recibiendo demasiadas solicitudes en este momento. "
                "Intente nuevamente en unos minutos."
            ),
        ) from exc

    temporarily_unavailable = (
        "503" in msg
        or "unavailable" in lowered
        or "service unavailable" in lowered
        or "high demand" in lowered
        or "demand spikes" in lowered
        or "try again later" in lowered
    )
    if temporarily_unavailable:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo de IA está experimentando alta demanda. "
                "Intente nuevamente en unos minutos."
            ),
        ) from exc

    raise HTTPException(
        status_code=502,
        detail=(
            "No pudimos completar el análisis con IA en este momento. "
            "Intente nuevamente en unos minutos."
        ),
    ) from exc


def _minimum_points_for(chart_type: str) -> int:
    return 3 if chart_type == "scatter" else 2


def _is_suggestion_usable(frame: pd.DataFrame, suggestion: ChartSuggestion) -> bool:
    try:
        series = aggregate_chart_dataframe(frame, suggestion.chart_type, suggestion.parameters)
    except ValueError:
        return False
    return len(series.data) >= _minimum_points_for(suggestion.chart_type)


def _build_fallback_suggestions(frame: pd.DataFrame) -> list[ChartSuggestion]:
    out: list[ChartSuggestion] = []
    numeric_cols = [str(c) for c in frame.select_dtypes(include=["number"]).columns]
    text_cols = [str(c) for c in frame.columns if str(c) not in numeric_cols]

    def add(candidate: ChartSuggestion):
        if any(
            s.chart_type == candidate.chart_type and s.parameters == candidate.parameters
            for s in out
        ):
            return
        if _is_suggestion_usable(frame, candidate):
            out.append(candidate)

    if text_cols:
        add(
            ChartSuggestion(
                title="Distribución por categoría principal",
                chart_type="pie",
                parameters={"category": text_cols[0]},
                insight=(
                    "La distribución permite identificar los segmentos con mayor peso relativo y "
                    "priorizar acciones en los grupos de mayor impacto."
                ),
            )
        )

    if text_cols:
        y_col = numeric_cols[0] if numeric_cols else text_cols[0]
        add(
            ChartSuggestion(
                title="Concentración por categoría",
                chart_type="bar",
                parameters={"x_axis": text_cols[0], "y_axis": y_col},
                insight=(
                    "Este análisis revela concentración por categoría y ayuda a enfocar recursos en "
                    "las entidades con mayor contribución."
                ),
            )
        )

    if len(numeric_cols) >= 2:
        add(
            ChartSuggestion(
                title="Relación entre métricas clave",
                chart_type="scatter",
                parameters={"x_axis": numeric_cols[0], "y_axis": numeric_cols[1]},
                insight=(
                    "La correlación entre ambas métricas ayuda a detectar patrones de rendimiento "
                    "y posibles oportunidades de optimización."
                ),
            )
        )

    if text_cols:
        y_col = numeric_cols[0] if numeric_cols else text_cols[0]
        add(
            ChartSuggestion(
                title="Comparativo de categorías secundarias",
                chart_type="bar",
                parameters={"x_axis": text_cols[min(1, len(text_cols) - 1)], "y_axis": y_col},
                insight=(
                    "Comparar categorías secundarias permite identificar desviaciones operativas y "
                    "ajustar decisiones comerciales con mayor precisión."
                ),
            )
        )

    if len(out) >= 3:
        return out[:5]
    return []


def _select_reliable_suggestions(
    frame: pd.DataFrame, llm_suggestions: list[ChartSuggestion]
) -> list[ChartSuggestion]:
    unique: dict[tuple[str, str], ChartSuggestion] = {}
    for s in llm_suggestions:
        key = (s.chart_type, str(sorted(s.parameters.items())))
        if key in unique:
            continue
        unique[key] = s
    valid = [s for s in unique.values() if _is_suggestion_usable(frame, s)]
    if len(valid) >= 3:
        return valid[:5]
    fallback = _build_fallback_suggestions(frame)
    combined: list[ChartSuggestion] = []
    seen: set[tuple[str, str]] = set()
    for s in [*valid, *fallback]:
        key = (s.chart_type, str(sorted(s.parameters.items())))
        if key in seen:
            continue
        seen.add(key)
        combined.append(s)
    return combined[:5]


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analizar archivo y generar sugerencias",
    description=(
        "Recibe un archivo tabular (`.csv` o `.xlsx`), construye su perfil, solicita entre 3 y 5 "
        "sugerencias de visualización con IA, genera metadata para cliente y persiste contenido en Supabase.\n\n"
        "Devuelve `upload_id` para consultas posteriores de series."
    ),
    responses={
        400: {"description": "Archivo inválido, vacío o con formato no soportado."},
        429: {"description": "Cuota/límite de Gemini alcanzado en los modelos probados."},
        502: {"description": "Fallo aguas arriba (LLM o persistencia en Supabase)."},
        503: {"description": "Proveedor IA temporalmente no disponible por alta demanda."},
    },
)
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
        _raise_user_friendly_llm_error(exc)

    suggestions = _select_reliable_suggestions(frame, suggestions)
    if len(suggestions) < 3:
        fallback_id = str(uuid.uuid4())
        logger.warning(
            "No reliable chart suggestions for upload %s: generated=%s",
            fallback_id,
            len(suggestions),
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "No se pudieron construir al menos 3 gráficas confiables con puntos suficientes "
                "para este archivo. Revise tipos de columna o cargue más registros válidos."
            ),
        )

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
