from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

ChartTypeLiteral = Literal["bar", "line", "pie", "scatter"]

SchemaColumnTypeLiteral = Literal["int", "float", "string", "date", "time", "datetime"]

ALLOWED_EXTENSIONS = frozenset({".csv", ".xlsx"})
ALLOWED_CHART_TYPES = frozenset({"bar", "line", "pie", "scatter"})

CANONICAL_SCHEMA_TYPES: tuple[str, ...] = ("int", "float", "string", "date", "time", "datetime")


class ChartSuggestion(BaseModel):
    title: str
    chart_type: ChartTypeLiteral
    parameters: dict[str, str] = Field(default_factory=dict)
    insight: str


class DatasetColumn(BaseModel):
    """Per-column typing hints for the data settings UI."""

    name: str
    pandas_dtype: str
    kind: Literal["datetime", "numeric", "text"]
    select_options: list[str]
    default_select: SchemaColumnTypeLiteral
    display_name: str | None = Field(
        default=None,
        description="Canonical visible label used across preview, mapping and IA hints.",
    )
    suggested_short_label: str | None = Field(
        default=None,
        description="Short Spanish label from IA for UI; parquet key remains `name`.",
    )


class AiColumnCorrection(BaseModel):
    """Structured refinement(s) derived from profiling heuristics for the IA insight card."""

    column: str
    target_type: SchemaColumnTypeLiteral
    suggested_header: str | None = None


class DatasetTypeDistribution(BaseModel):
    strings: int
    numerics: int
    datetimes: int


class DataframeClientDataset(BaseModel):
    row_count: int
    column_count: int
    sample_tag: str
    columns: list[DatasetColumn]
    preview_rows: list[dict[str, str]]
    null_cells: int
    duplicate_rows: int
    memory_mb: float
    numeric_skew: float | None
    integrity_percent: float
    ai_hint: str
    ai_corrections: list[AiColumnCorrection] = Field(default_factory=list)
    type_distribution: DatasetTypeDistribution


class AnalyzeResponse(BaseModel):
    upload_id: str
    suggestions: Annotated[list[ChartSuggestion], Field(min_length=3, max_length=5)]
    dataset: DataframeClientDataset


class ChartSeriesRequest(BaseModel):
    upload_id: str
    chart_type: ChartTypeLiteral
    parameters: dict[str, str]


class ChartSeriesResponse(BaseModel):
    chart_type: ChartTypeLiteral
    data: list[dict[str, Any]]
