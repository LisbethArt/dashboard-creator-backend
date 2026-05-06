from functools import lru_cache
from typing import Annotated

import httpx
from fastapi import Depends
from supabase import Client, ClientOptions, create_client

from app.config import Settings, get_settings


@lru_cache
def _cached_supabase(url: str, key: str) -> Client:
    http_client = httpx.Client(
        http2=False,
        timeout=httpx.Timeout(120.0, connect=30.0),
        follow_redirects=True,
    )
    options = ClientOptions(httpx_client=http_client)
    return create_client(url, key, options)


def get_supabase(settings: Annotated[Settings, Depends(get_settings)]) -> Client:
    return _cached_supabase(settings.supabase_url, settings.supabase_service_role_key)
