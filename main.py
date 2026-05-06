"""Dashboard Creator HTTP API. Run with ``uvicorn main:app`` from this directory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze, charts

settings = get_settings()

app = FastAPI(title="Dashboard Creator API", version="1.0.0")

_explicit = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
_cors_kw: dict = {
    "allow_credentials": False,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if _explicit:
    app.add_middleware(CORSMiddleware, allow_origins=_explicit, **_cors_kw)
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        **_cors_kw,
    )

app.include_router(analyze.router, prefix="/api")
app.include_router(charts.router, prefix="/api")
