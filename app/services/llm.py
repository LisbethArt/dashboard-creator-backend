import json
import logging
import math
import re
from typing import Any

from google import genai

from app.config import Settings
from app.schemas import ALLOWED_CHART_TYPES, ChartSuggestion
from app.services.profile import DataFrameProfileText

logger = logging.getLogger(__name__)

# Fallback order when the active model hits quota or rate limits: cheaper first, then stronger models.
GEMINI_MODEL_FALLBACK_CHAIN: tuple[str, ...] = (
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
)


class GeminiQuotaOrRateLimit(Exception):
    """Raised when the API signals exhaustion so the caller can try another model id."""


class GeminiModelUnavailable(Exception):
    """Raised when the model id is unknown or not enabled for this key; try another id."""


def _is_quota_or_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if exc.__class__.__name__.lower() in ("resourceexhausted", "toomanyrequests"):
        return True
    needles = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "too many requests",
    )
    return any(n in msg for n in needles)


def _is_model_not_available(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "404" in msg
        or "not found" in msg
        or "invalid model" in msg
        or "does not exist" in msg
        or "unknown model" in msg
    )


def _unique_model_order(primary: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in (primary.strip(), *GEMINI_MODEL_FALLBACK_CHAIN):
        if not raw or raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return ordered


def _response_text(response: Any) -> str:
    direct = getattr(response, "text", None)
    if direct:
        return str(direct).strip()
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if block:
        raw = block.group(1)
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("Model output is not a JSON array")
    return parsed


def _normalize_chart_type(value: str) -> str:
    v = (value or "").strip().lower()
    mapping = {
        "bar_chart": "bar",
        "line_chart": "line",
        "pie_chart": "pie",
        "scatter_plot": "scatter",
    }
    return mapping.get(v, v)


def _parse_suggestions_payload(raw: str) -> list[ChartSuggestion]:
    items = _extract_json_array(raw)
    out: list[ChartSuggestion] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ct = _normalize_chart_type(str(item.get("chart_type", "")))
        if ct not in ALLOWED_CHART_TYPES:
            continue
        params = item.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        params = {str(k): str(v) for k, v in params.items()}
        out.append(
            ChartSuggestion(
                title=str(item.get("title", "Untitled chart")),
                chart_type=ct,  # type: ignore[arg-type]
                parameters=params,
                insight=str(item.get("insight", "")),
            )
        )
    return out


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw)
    if fenced:
        snippet = fenced.group(1)
    else:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object in model response")
        snippet = raw[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON is not an object")
    return parsed


def _coerce_label_map(payload: dict[str, Any], valid: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw_val in payload.items():
        ks = str(key).strip()
        if ks not in valid:
            continue
        label = str(raw_val).strip()
        label = label.replace("\n", " ").strip().strip("\"'«»")
        if not label:
            continue
        out[ks] = label[:96]
    return out


def generate_column_short_labels(
    profile: DataFrameProfileText,
    valid_columns: list[str],
    settings: Settings,
) -> dict[str, str]:
    """
    Asks Gemini for concise Spanish labels keyed by technical parquet / pandas column identifiers.

    Returns a partial mapping on success; callers merge onto ``DatasetColumn`` without blocking analyze.
    """
    keys = frozenset(valid_columns)
    if not keys:
        return {}

    labeling_rules = (
        "You are a data librarian localising survey and business tables.\n"
        "Task: build a SHORT display name per column key for dashboards (Spanish).\n"
        "\n"
        "Rules:\n"
        "- JSON object ONLY — no prose, no markdown fences needed but valid JSON mandatory.\n"
        "- KEYS must be IDENTICAL UTF-8 strings to the identifiers listed below (quote-sensitive), including "
        "'Unnamed: 0', 'Unnamed: 1', etc.\n"
        "- Values: 2–5 words Title Case Español, max 42 characters, neutral tone.\n"
        "- Long questionnaire headers ('¿Cuál es su nivel educativo?') → distill (e.g., 'Nivel educativo'; "
        "'Nombre' for full-name fields; 'Modalidad trabajo' where appropriate).\n"
        "- Columns named Unnamed or generic placeholders: derive from sample_cells_first_rows and "
        "categorical_top_values; if samples look like headings (Cantidad, Categoría…) treat as descriptors.\n"
        "- Mostly qualitative text with categories is NORMAL — propose labels that hint the dimension measured.\n"
        "- Never copy an entire paragraph; never include question marks unless part of kept acronym edge case "
        "(avoid).\n"
        "- Emit one JSON entry per identifier listed under column_names_dtypes_nunique in the profile "
        "(the quoted segment before `: dtype=`). Omit keys only if unsure—coverage should be maximal.\n"
        "\n"
        "Return one JSON object mapping each technical column identifier → short Spanish label.\n"
    )

    payload = labeling_rules + "\nFull dataset profile for context:\n" + profile.text

    client = genai.Client(api_key=settings.gemini_api_key)
    threshold = max(1, math.ceil(len(keys) * 0.42))
    floor_partial = max(1, math.ceil(len(keys) * 0.22))

    try:
        models_to_try = _unique_model_order(settings.gemini_model)
        last_quota: GeminiQuotaOrRateLimit | None = None

        def run_prompt(model_name: str, text: str) -> str:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=text,
                )
            except Exception as exc:
                if _is_model_not_available(exc):
                    logger.warning(
                        "Gemini model %s is unavailable for this project or key; skipping",
                        model_name,
                    )
                    raise GeminiModelUnavailable(str(exc)) from exc
                if _is_quota_or_rate_limit(exc):
                    raise GeminiQuotaOrRateLimit(str(exc)) from exc
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            out = _response_text(response)
            if not out:
                raise RuntimeError("Empty Gemini response")
            return out

        def try_labels(model_name: str) -> dict[str, str]:
            first = run_prompt(model_name, payload)
            mapped: dict[str, str] = {}
            try:
                mapped = _coerce_label_map(_extract_json_object(first), keys)
            except (json.JSONDecodeError, ValueError):
                mapped = {}

            if len(mapped) >= threshold:
                return mapped
            if len(mapped) >= floor_partial:
                logger.info(
                    "Partial IA column labels on first pass (%s of %s keys)",
                    len(mapped),
                    len(keys),
                )
                return mapped

            repair = (
                "Your previous answer was not a single JSON object or missed keys. Reply with ONLY a JSON object "
                "mapping each ALLOWED_KEYS entry to a short Spanish label. Keys must match exactly. "
                "Profile:\n\n"
                f"{profile.text}\n\nBad output (truncated):\n{first[:3500]}"
            )
            second = run_prompt(model_name, repair)
            try:
                mapped = _coerce_label_map(_extract_json_object(second), keys)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"Label repair JSON invalid: {exc}") from exc
            if len(mapped) >= threshold:
                return mapped
            if len(mapped) >= floor_partial:
                logger.info(
                    "Applying partial IA column labels (%s of %s keys)",
                    len(mapped),
                    len(keys),
                )
                return mapped
            raise RuntimeError("Model returned too few column labels after repair")

        for model_name in models_to_try:
            try:
                result = try_labels(model_name)
                if model_name != settings.gemini_model.strip():
                    logger.info(
                        "Column labels obtained with fallback Gemini model %s (configured: %s)",
                        model_name,
                        settings.gemini_model,
                    )
                return result
            except GeminiModelUnavailable:
                continue
            except GeminiQuotaOrRateLimit as exc:
                last_quota = exc
                logger.warning(
                    "Gemini quota or rate limit on model %s while labeling columns; trying next",
                    model_name,
                )
                continue
            except RuntimeError as exc:
                logger.warning("Column label attempt failed on %s: %s", model_name, exc)
                continue

        if last_quota:
            logger.warning("All Gemini models hit quota for column labels; continuing without hints")
        return {}
    except Exception as exc:
        logger.warning("Column label pipeline error: %s", exc)
        return {}
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def generate_chart_suggestions(profile: DataFrameProfileText, settings: Settings) -> list[ChartSuggestion]:
    """
    Calls Gemini with a strict analyst prompt and returns 3–5 validated ``ChartSuggestion`` records.

    Uses the Google Gen AI SDK (``google-genai``). The model must answer with a JSON array only.
    If parsing fails, a second attempt is made with an explicit repair instruction; otherwise a
    ``RuntimeError`` is raised for HTTP mapping.

    On quota or rate-limit errors, automatically retries the same pipeline with other model ids
    (economical options first, then heavier models as a last resort).
    """
    chart_viz_rules = (
        "Visualization fit and legibility (mandatory — crowded or misleading charts are failures):\n"
        "0) Preflight: For every proposed x_axis, category, label, or grouping column, locate that column in "
        "'column_names_dtypes_nunique' and read its nunique. Never assume a low count; never use bar/pie/line "
        "with a nominal axis when nunique is unknown from the profile excerpt.\n"
        "1) bar (vertical categories on X): ONLY when the chosen x_axis has nunique ≤ 12 AND labels are short enough "
        "to stay distinct. If nunique is 13–40, do NOT use bar on that entity column; instead: (a) pivot to a "
        "coarser column in the profile with nunique ≤ 12 (region, family, bucket, channel, segment, week, month), "
        "(b) use line with a temporal/ordered x if such a column exists, or (c) use scatter with two different "
        "numeric measures from numeric_describe that answer a related question. If no coarser column exists, pick "
        "another analytic angle (trend, correlation, distribution of one numeric) instead of forcing many thin bars.\n"
        "   High-cardinality entity roles (often high nunique even when dtype is int/float): product, sku, item, "
        "article, modelo, cliente, usuario, orden, invoice, transaction, id — treat them like labels: bar on these "
        "is forbidden unless their nunique ≤ 12.\n"
        "2) pie / part-of-whole: ONLY when nunique ≤ 8, slices encode shares of ONE total, categories are exhaustive "
        "or clearly labeled “Otros/complemento”. Do not use pie for ranking many suppliers, products, or people.\n"
        "3) line: x_axis MUST be time-like or strictly ordered (dates, timestamps, año-mes, week index, sequential "
        "period). Forbidden for unordered SKUs/strings with many buckets. Prefer resampling/grouping dates to "
        "week/month if daily points explode.\n"
        "4) scatter: both x_axis and y_axis MUST be numeric columns present under numeric_describe (continuous or "
        "count-like). Never pair category vs number in scatter — use aggregated bar instead only if cardinality allows.\n"
        "5) Scenario playbook (datasets vary randomly; pick what the profile supports): many rows + few buckets → "
        "bar/pie when thresholds pass; longitudinal signal → line; relate two KPIs → scatter; "
        "many entities / long tail: avoid per-entity categorical charts entirely and move to timing, coarse segments, "
        "or numeric-numeric insight; quasi-duplicate keys → mention aggregation caveat in insight.\n"
        "6) Portfolio: Across the 3–5 suggestions vary chart_type when reasonable so the dashboard is not mostly "
        "unreadable bars copied with different titles.\n"
    )
    system_rules = (
        "You are an expert data analyst. Given a dataset profile, propose 3 to 5 chart ideas.\n"
        "Each idea must highlight a non-trivial pattern or relationship users should explore.\n"
        + chart_viz_rules
        + "\nReply with a JSON array ONLY. No markdown, no prose outside JSON.\n"
        "Each array element is an object with keys: title (string), chart_type (one of bar, line, pie, scatter), "
        "parameters (object mapping axis or role keys to existing column names from the profile), "
        "insight (short analytic paragraph in Spanish).\n"
        "CRITICAL parameters rule: Every column referenced in parameters must be spelled EXACTLY as in "
        "column_names_dtypes_nunique—including names like 'Unnamed: 0', spaces, punctuation, accents. Never invent "
        "column names from sample text only; qualitative text fields with moderate nunique are valid for bars/pie.\n"
        'Example element (valid only when region has nunique ≤ 12 in the profile): '
        '{"title":"Sales by Region","chart_type":"bar","parameters":'
        '{"x_axis":"region","y_axis":"revenue"},"insight":"Norte domina..."}\n'
    )

    payload = system_rules + "\nDataset profile:\n" + profile.text

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        models_to_try = _unique_model_order(settings.gemini_model)
        last_quota: GeminiQuotaOrRateLimit | None = None

        def run_prompt(model_name: str, text: str) -> str:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=text,
                )
            except Exception as exc:
                if _is_model_not_available(exc):
                    logger.warning(
                        "Gemini model %s is unavailable for this project or key; skipping",
                        model_name,
                    )
                    raise GeminiModelUnavailable(str(exc)) from exc
                if _is_quota_or_rate_limit(exc):
                    raise GeminiQuotaOrRateLimit(str(exc)) from exc
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
            out = _response_text(response)
            if not out:
                raise RuntimeError("Empty Gemini response")
            return out

        def try_pipeline(model_name: str) -> list[ChartSuggestion]:
            first = run_prompt(model_name, payload)
            try:
                suggestions = _parse_suggestions_payload(first)
            except (json.JSONDecodeError, ValueError):
                suggestions = []

            if 3 <= len(suggestions) <= 5:
                return suggestions

            repair = (
                "Your previous answer was invalid or had wrong length. Respond again with ONLY a JSON array "
                "containing between 3 and 5 objects. Keys per object: title, chart_type, parameters, insight. "
                "Chart_type must be bar, line, pie, or scatter. parameters must reference real columns from the profile. "
                "Enforce bar chart x_axis columns with nunique ≤ 12 read from column_names_dtypes_nunique; pie label "
                "columns nunique ≤ 8; scatter only numeric_describe numeric pairs; line only ordered/date-like x axes. "
                "Do not propose vertical bar charts where each SKU/product/client id maps to its own strip when "
                "nunique is high — pick coarser grouping, temporal, or scatter alternatives. Dataset profile:\n\n"
                f"{profile.text}"
            )
            second = run_prompt(model_name, repair + "\n\nInvalid prior output:\n" + first[:4000])
            try:
                suggestions = _parse_suggestions_payload(second)
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"Second model output was not valid JSON: {exc}") from exc
            if not (3 <= len(suggestions) <= 5):
                raise RuntimeError("Model did not return 3–5 valid chart suggestions")
            return suggestions

        for model_name in models_to_try:
            try:
                result = try_pipeline(model_name)
                if model_name != settings.gemini_model.strip():
                    logger.info(
                        "Chart suggestions obtained with fallback Gemini model %s (configured: %s)",
                        model_name,
                        settings.gemini_model,
                    )
                return result
            except GeminiModelUnavailable:
                continue
            except GeminiQuotaOrRateLimit as exc:
                last_quota = exc
                logger.warning(
                    "Gemini quota or rate limit on model %s; trying next fallback if available",
                    model_name,
                )
                continue

        if last_quota:
            tried = ", ".join(models_to_try)
            raise RuntimeError(
                "Gemini quota or rate limit on all attempted models "
                f"({tried}). Wait briefly or review limits in Google AI Studio."
            ) from last_quota

        raise RuntimeError(
            "None of the configured Gemini model IDs worked for this API key "
            f"(tried: {', '.join(models_to_try)}). Check model names in Google AI documentation and GEMINI_MODEL in .env."
        )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"LLM pipeline failed: {exc}") from exc
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
