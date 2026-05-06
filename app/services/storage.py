import io
import uuid
from typing import Any

import pandas as pd
from supabase import Client

from app.schemas import ChartSuggestion


def _sanitize_for_pyarrow_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns that PyArrow cannot infer as a single Parquet physical type.

    Spreadsheets often load mixed numbers and labels into ``object`` columns (e.g. ``Estado``),
    which raises conversion errors on ``DataFrame.to_parquet``.
    """
    out = df.copy()
    for col in out.select_dtypes(include="category").columns:
        out[col] = out[col].astype("string")
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].map(
            lambda v: pd.NA if pd.isna(v) else str(v).replace("\x00", "")
        ).astype(pd.StringDtype())
    return out


def new_upload_id() -> str:
    return str(uuid.uuid4())


def storage_object_paths(upload_id: str, extension: str) -> tuple[str, str]:
    normalized = extension if extension.startswith(".") else f".{extension}"
    original_key = f"{upload_id}/original{normalized}"
    parquet_key = f"{upload_id}/dataset.parquet"
    return original_key, parquet_key


def upload_file(
    client: Client,
    bucket: str,
    object_path: str,
    data: bytes,
    content_type: str,
) -> None:
    client.storage.from_(bucket).upload(
        object_path,
        data,
        file_options={"content-type": content_type, "upsert": "true"},
    )


def download_bytes(client: Client, bucket: str, object_path: str) -> bytes:
    return client.storage.from_(bucket).download(object_path)


def persist_upload_bundle(
    client: Client,
    bucket: str,
    upload_id: str,
    original_key: str,
    parquet_key: str,
    dataframe: pd.DataFrame,
    original_bytes: bytes,
    original_media_type: str,
    suggestions: list[ChartSuggestion],
) -> None:
    """
    Stores the raw file, derived Parquet dataset, and suggestion metadata in Supabase.

    The Parquet object enables efficient repeated chart aggregation without re-parsing spreadsheets.
    """
    buffer = io.BytesIO()
    safe_frame = _sanitize_for_pyarrow_parquet(dataframe)
    safe_frame.to_parquet(buffer, index=False, engine="pyarrow")
    parquet_bytes = buffer.getvalue()
    upload_file(client, bucket, original_key, original_bytes, original_media_type)
    upload_file(
        client,
        bucket,
        parquet_key,
        parquet_bytes,
        "application/vnd.apache.parquet",
    )
    payload: dict[str, Any] = {
        "id": upload_id,
        "original_path": original_key,
        "parquet_path": parquet_key,
        "suggestions": [item.model_dump(mode="json") for item in suggestions],
    }
    client.table("uploads").insert(payload).execute()


def fetch_upload_row(client: Client, upload_id: str) -> dict[str, Any] | None:
    response = client.table("uploads").select("*").eq("id", upload_id).limit(1).execute()
    rows = response.data or []
    return rows[0] if rows else None


def load_dataframe_from_parquet(client: Client, bucket: str, parquet_key: str) -> pd.DataFrame:
    raw = download_bytes(client, bucket, parquet_key)
    return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
