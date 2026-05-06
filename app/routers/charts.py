from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from app.config import Settings, get_settings
from app.deps import get_supabase
from app.schemas import ChartSeriesRequest, ChartSeriesResponse
from app.services.aggregation import aggregate_chart_dataframe
from app.services.storage import fetch_upload_row, load_dataframe_from_parquet

router = APIRouter(tags=["charts"])


@router.post("/charts/series", response_model=ChartSeriesResponse)
def build_chart_series(
    body: ChartSeriesRequest,
    settings: Settings = Depends(get_settings),
    client: Client = Depends(get_supabase),
) -> ChartSeriesResponse:
    record = fetch_upload_row(client, body.upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")

    parquet_key = record.get("parquet_path")
    if not parquet_key or not isinstance(parquet_key, str):
        raise HTTPException(status_code=500, detail="Upload metadata missing parquet path")

    try:
        frame = load_dataframe_from_parquet(client, settings.supabase_bucket, parquet_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load dataset: {exc}") from exc

    try:
        return aggregate_chart_dataframe(frame, body.chart_type, body.parameters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
