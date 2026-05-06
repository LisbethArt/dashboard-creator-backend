"""Dashboard Creator HTTP API. Run with ``uvicorn main:app`` from this directory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import analyze, charts

settings = get_settings()

OPENAPI_TAGS = [
    {
        "name": "analyze",
        "description": (
            "Carga archivos tabulares (.csv/.xlsx), perfila el dataset, solicita sugerencias de "
            "gráficas con IA y persiste artefactos en Supabase."
        ),
    },
    {
        "name": "charts",
        "description": (
            "Construye series agregadas para visualización leyendo el parquet persistido por upload_id."
        ),
    },
]

app = FastAPI(
    title="Dashboard Creator API",
    description=(
        "API para el flujo completo del Dashboard Creator.\n\n"
        "Incluye:\n"
        "- Ingesta de archivos CSV/XLSX.\n"
        "- Perfilado de columnas y vista previa de datos.\n"
        "- Sugerencias automáticas de gráficos usando Gemini.\n"
        "- Persistencia de archivo original + parquet + metadatos en Supabase.\n"
        "- Generación de series agregadas para `bar`, `line`, `pie` y `scatter`.\n\n"
        "Notas:\n"
        "- `upload_id` identifica cada carga y se usa para consultar series posteriormente.\n"
        "- Los errores 429/503 en análisis suelen venir de límites o saturación temporal del proveedor IA."
    ),
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "docExpansion": "full",
        "displayRequestDuration": True,
        "operationsSorter": "method",
        "tagsSorter": "alpha",
    },
)

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
