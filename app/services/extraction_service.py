"""
EXTRACTION SERVICE
-------------------
Owns everything from "normalized OCR tokens" to "validated JSON fields".
If the LLM changes tomorrow (different model / different provider), only
`call_llm` needs to change — the prompt-building and validation logic stays.

Public API used by the router:
    build_ocr_context(ocr_result) -> str
    call_llm(context, filename) -> dict          (raw parsed JSON from model)
    extract_fields(ocr_result) -> ExtractionResult   (full flow incl. retries)
"""
import json
import re
from collections import defaultdict
from typing import Dict, Any, List

import httpx

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.schemas import OCRResult, ExtractionResult, ExtractedField

logger = get_logger(__name__)

_client = None


def _get_client() -> Any:
    global _client
    if _client is None:
        if anthropic is None:
            raise RuntimeError("anthropic package is not installed")
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY or None)
    return _client


def _call_openai_compatible_llm(user_prompt: str) -> str:
    api_base = (settings.LLM_API_BASE_URL or "").rstrip("/")
    if not api_base:
        raise RuntimeError("LLM_API_BASE_URL is not configured")

    url = f"{api_base}/chat/completions"
    headers = {}
    if settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

    payload = {
        "model": settings.LLM_MODEL,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=180.0)
    response.raise_for_status()
    data = response.json()

    if not data.get("choices"):
        raise ValueError(f"Unexpected LLM response: {data}")

    message = data["choices"][0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        raise ValueError(f"Unexpected LLM content structure: {content}")
    return content


# --------------------------------------------------------------------------
# Step 1: turn OCR tokens into a compact, structured text context for the LLM
# --------------------------------------------------------------------------
def build_ocr_context(ocr_result: OCRResult) -> str:
    """
    Groups tokens by line_id (already computed geometrically by the OCR
    service) and renders a plain-text "layout preview" so the LLM can reason
    about which text is a label vs. a value, without needing raw pixel coords.
    """
    by_line: Dict[int, List] = defaultdict(list)
    for tok in ocr_result.tokens:
        key = tok.line_id if tok.line_id is not None else -1
        by_line[key].append(tok)

    lines_out = []
    for line_id in sorted(by_line.keys()):
        toks = sorted(by_line[line_id], key=lambda t: t.bbox.x1)
        line_text = "  |  ".join(f"{t.text}" for t in toks)
        avg_conf = sum(t.confidence for t in toks) / len(toks)
        lines_out.append(f"[line {line_id}, conf={avg_conf:.2f}] {line_text}")

    return "\n".join(lines_out)


# --------------------------------------------------------------------------
# Step 2: call the LLM with strict JSON-only instructions
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a form-data extraction engine. You will be given OCR
output from a scanned form or document, organized line-by-line (OCR noise and
minor misreads are possible).

Your job: identify every label/value pair, checkbox state, and table in the
document, and return ONLY a single JSON object — no markdown fences, no
commentary, no explanation — matching exactly this shape:

{
  "fields": [
    {
      "label": "<normalized snake_case label>",
      "value": "<string, list of strings, or object for tables>",
      "confidence": <float 0-1>,
      "field_type": "text" | "multiline" | "checkbox" | "table"
    }
  ],
  "warnings": ["<any ambiguity, low-confidence read, or missing field you noticed>"]
}

Rules:
- Merge multi-line values that clearly belong to one label into a single field
  with field_type "multiline" and value as a list of strings in order.
- For checkboxes, value must be true or false (as a JSON boolean-like string
  "true"/"false" is NOT acceptable — use native booleans).
- For tables, value must be an object: {"headers": [...], "rows": [[...], ...]}.
- If a label has no discernible value, still include it with value null.
- Never invent data that isn't present in the OCR text.
- Output must be valid JSON and nothing else.
"""


def _extract_json_block(text: str) -> str:
    """Strips markdown fences if the model adds them despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def call_llm(context: str, filename: str) -> Dict[str, Any]:
    user_prompt = f"Filename: {filename}\n\nOCR layout (line-grouped):\n{context}"

    if settings.LLM_API_BASE_URL:
        raw_text = _call_openai_compatible_llm(user_prompt)
    else:
        client = _get_client()
        response = client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )

    cleaned = _extract_json_block(raw_text)
    return json.loads(cleaned)  # raises JSONDecodeError if invalid -> triggers retry


# --------------------------------------------------------------------------
# Step 3: validate + retry loop, then convert to ExtractionResult
# --------------------------------------------------------------------------
def _normalize_field_value(raw_value: Any, field_type: str) -> Any:
    if raw_value is None:
        return None
    if field_type == "checkbox":
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        if isinstance(raw_value, list):
            return [
                {
                    "label": item.get("label", "") if isinstance(item, dict) else str(item),
                    "checked": bool(item.get("checked", False)) if isinstance(item, dict) else False,
                }
                for item in raw_value
            ]
        return bool(raw_value)
    if field_type == "table":
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, list):
            return {"headers": [], "rows": raw_value}
        return {"headers": [], "rows": []}
    if isinstance(raw_value, list):
        normalized = []
        for item in raw_value:
            if isinstance(item, dict):
                if "label" in item:
                    normalized.append(str(item["label"]))
                elif "value" in item:
                    normalized.append(str(item["value"]))
                else:
                    normalized.append(str(item))
            else:
                normalized.append(str(item))
        return normalized
    if isinstance(raw_value, dict):
        return str(raw_value)
    return str(raw_value)


def _validate_shape(parsed: Dict[str, Any]) -> List[str]:
    """Lightweight schema check beyond what Pydantic enforces on ExtractedField."""
    problems = []
    if "fields" not in parsed or not isinstance(parsed["fields"], list):
        problems.append("Missing or invalid 'fields' array")
        return problems
    for i, f in enumerate(parsed["fields"]):
        if not isinstance(f, dict):
            problems.append(f"Field {i} is not an object")
            continue
        if "label" not in f:
            problems.append(f"Field {i} missing 'label'")
        if "field_type" not in f:
            problems.append(f"Field {i} missing 'field_type'")
    return problems


def extract_fields(ocr_result: OCRResult) -> ExtractionResult:
    context = build_ocr_context(ocr_result)
    warnings: List[str] = []
    last_error = None

    for attempt in range(1, settings.LLM_MAX_RETRIES + 2):  # +1 initial +N retries
        try:
            parsed = call_llm(context, ocr_result.filename)
            shape_problems = _validate_shape(parsed)
            if shape_problems:
                raise ValueError(f"Schema validation failed: {shape_problems}")

            fields = []
            for f in parsed["fields"]:
                field_type = str(f.get("field_type", "text"))
                fields.append(
                    ExtractedField(
                        label=str(f.get("label", "unknown")),
                        value=_normalize_field_value(f.get("value"), field_type),
                        confidence=float(f.get("confidence", 0.8)),
                        field_type=field_type,
                    )
                )
            warnings.extend(parsed.get("warnings", []))

            return ExtractionResult(
                filename=ocr_result.filename,
                page_count=ocr_result.page_count,
                fields=fields,
                raw_token_count=len(ocr_result.tokens),
                llm_attempts=attempt,
                warnings=warnings,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = str(e)
            logger.warning(f"LLM output invalid on attempt {attempt}: {last_error}")
            continue

    # all attempts exhausted -> return empty result with warning, don't crash the API
    warnings.append(f"LLM failed to produce valid JSON after retries: {last_error}")
    return ExtractionResult(
        filename=ocr_result.filename,
        page_count=ocr_result.page_count,
        fields=[],
        raw_token_count=len(ocr_result.tokens),
        llm_attempts=settings.LLM_MAX_RETRIES + 1,
        warnings=warnings,
    )
