import re

import pandas as pd

from app.schemas import ChartSeriesResponse

INVALID_AXIS_LABELS = frozenset({"nan", "nat", "none", ""})

# Max categories for bar / categorical-like line grouping — aligns with LLM legibility rules.
BAR_LINE_CATEGORY_CAP = 16

# Pie slices capped so radial legends remain tractable on dashboards.
PIE_SLICE_CAP = 10


def _pick_column(parameters: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = parameters.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _normalized_column_token(name: object) -> str:
    """Lowercase fingerprint with collapsed whitespace so LLM hints match parquet headers."""
    return re.sub(r"\s+", " ", str(name).strip()).casefold()


def _resolve_dataframe_column(df: pd.DataFrame, requested: str | None, role: str) -> str:
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError(f"Missing {role} column hint in parameters")

    hint = requested.strip()
    columns = df.columns.to_list()

    exact = [c for c in columns if str(c).strip() == hint]
    if len(exact) == 1:
        return exact[0]

    hint_key = _normalized_column_token(hint)
    matches = [c for c in columns if _normalized_column_token(c) == hint_key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous {role} column {requested!r}; matches {[str(c) for c in matches]}"
        )

    listed = ", ".join(repr(str(c)) for c in columns)
    raise ValueError(
        f"No column matching {requested!r} for {role}. Available columns: [{listed}]"
    )


def _sanitize_label_for_output(label: object) -> bool:
    text = str(label).strip().lower()
    return text not in INVALID_AXIS_LABELS


def _prepare_bar_line_frame(
    df: pd.DataFrame, x_col: str, y_col: str, chart_type: str
) -> pd.DataFrame:
    work = df[[x_col, y_col]].copy()
    work = work.dropna(subset=[x_col, y_col])
    if chart_type == "line":
        parsed = pd.to_datetime(work[x_col], errors="coerce")
        work = work.assign(_group_key=parsed)
        work = work.dropna(subset=["_group_key"])
        work["_group_key"] = work["_group_key"].dt.strftime("%Y-%m-%d")
        return work[["_group_key", y_col]]
    stripped = work[x_col].map(lambda x: str(x).strip() if pd.notna(x) else "")
    invalid = stripped.str.lower().isin(INVALID_AXIS_LABELS)
    work = work.loc[~invalid].copy()
    work["_group_key"] = stripped.loc[~invalid].astype(str)
    return work[["_group_key", y_col]]


def aggregate_chart_dataframe(
    df: pd.DataFrame,
    chart_type: str,
    parameters: dict[str, str],
) -> ChartSeriesResponse:
    """
    Aggregates rows from ``df`` according to ``chart_type`` and column hints in ``parameters``.

    Returned ``data`` rows use Recharts-friendly keys: ``name`` and ``value`` for categorical charts;
    ``x`` and ``y`` for scatter plots. Nominal/bar payload size is capped for dashboard readability.
    """
    if df.empty:
        return ChartSeriesResponse(chart_type=chart_type, data=[])

    if chart_type in {"bar", "line"}:
        x_col = _resolve_dataframe_column(
            df, _pick_column(parameters, "x_axis", "x", "category", "group"), "x_axis"
        )
        y_col = _resolve_dataframe_column(df, _pick_column(parameters, "y_axis", "y", "value"), "y_axis")
        work = _prepare_bar_line_frame(df, x_col, y_col, chart_type)
        if work.empty:
            return ChartSeriesResponse(chart_type=chart_type, data=[])
        gkey = "_group_key"
        if pd.api.types.is_numeric_dtype(work[y_col]):
            grouped = work.groupby(gkey, dropna=False)[y_col].sum(min_count=1)
            if chart_type == "line":
                grouped = grouped.sort_index(kind="stable").tail(366)
            else:
                grouped = grouped.sort_values(ascending=False).head(BAR_LINE_CATEGORY_CAP)
            data = [
                {"name": str(idx), "value": float(val)}
                for idx, val in grouped.items()
                if _sanitize_label_for_output(idx) and pd.notna(val)
            ]
        else:
            counts = work.groupby(gkey, dropna=False).size()
            counts = counts.sort_values(ascending=False).head(BAR_LINE_CATEGORY_CAP)
            data = [
                {"name": str(idx), "value": float(val)}
                for idx, val in counts.items()
                if _sanitize_label_for_output(idx) and pd.notna(val)
            ]
        return ChartSeriesResponse(chart_type=chart_type, data=data)

    if chart_type == "pie":
        label_col = _resolve_dataframe_column(
            df, _pick_column(parameters, "category", "label", "x_axis", "x"), "label/category"
        )
        value_hint = _pick_column(parameters, "value", "y_axis", "y")
        if value_hint:
            value_col = _resolve_dataframe_column(df, value_hint, "value")
            work = df[[label_col, value_col]].copy()
            work = work.dropna(subset=[label_col])
            lbl = work[label_col].map(lambda x: str(x).strip() if pd.notna(x) else "")
            mask = ~lbl.str.lower().isin(INVALID_AXIS_LABELS)
            work = work.loc[mask].copy()
            work["_pk"] = lbl.loc[mask].astype(str)
            if pd.api.types.is_numeric_dtype(work[value_col]):
                work = work.dropna(subset=[value_col])
                grouped = (
                    work.groupby("_pk", dropna=False)[value_col].sum(min_count=1).sort_values(ascending=False).head(
                        PIE_SLICE_CAP
                    )
                )
                data = [
                    {"name": str(k), "value": float(v)}
                    for k, v in grouped.items()
                    if _sanitize_label_for_output(k) and pd.notna(v)
                ]
            else:
                counts = work.groupby("_pk", dropna=False).size().sort_values(ascending=False).head(PIE_SLICE_CAP)
                data = [
                    {"name": str(k), "value": float(v)}
                    for k, v in counts.items()
                    if _sanitize_label_for_output(k)
                ]
            return ChartSeriesResponse(chart_type=chart_type, data=data)
        vc = df[label_col].dropna()
        vc = vc.map(lambda x: str(x).strip() if pd.notna(x) else "")
        vc = vc[~vc.str.lower().isin(INVALID_AXIS_LABELS)]
        counts = vc.value_counts().head(PIE_SLICE_CAP)
        data = [
            {"name": str(k), "value": float(v)}
            for k, v in counts.items()
            if _sanitize_label_for_output(k)
        ]
        return ChartSeriesResponse(chart_type=chart_type, data=data)

    if chart_type == "scatter":
        x_col = _resolve_dataframe_column(df, _pick_column(parameters, "x_axis", "x"), "x_axis")
        y_col = _resolve_dataframe_column(df, _pick_column(parameters, "y_axis", "y"), "y_axis")
        sample = df[[x_col, y_col]].copy()
        sample[x_col] = pd.to_numeric(sample[x_col], errors="coerce")
        sample[y_col] = pd.to_numeric(sample[y_col], errors="coerce")
        sample = sample.dropna()
        if sample.empty:
            raise ValueError("scatter charts need numeric x_axis and y_axis columns")
        if len(sample) > 500:
            sample = sample.sample(500, random_state=42)
        data = [
            {"x": float(row[x_col]), "y": float(row[y_col])}
            for _, row in sample.iterrows()
        ]
        return ChartSeriesResponse(chart_type=chart_type, data=data)

    raise ValueError(f"Unsupported chart type {chart_type}")
