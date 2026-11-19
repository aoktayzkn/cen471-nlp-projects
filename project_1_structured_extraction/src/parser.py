"""Parser and validation module with strict schema verification and normalization."""

import json
import re
from typing import Any, Dict, Optional, Tuple

# ==============================================================================
# BASELINE VALIDATORS (v1) - [ARCHIVED]
# ==============================================================================
# VALID_SENTIMENTS_V1 = {"positive", "negative", "neutral"}

# ==============================================================================
# OPTIMIZED VALIDATORS (v2) - [ACTIVE]
# ==============================================================================
VALID_SENTIMENTS_V2 = {"positive", "negative"}

VALID_CATEGORIES = {
    "action",
    "comedy",
    "drama",
    "horror",
    "sci-fi",
    "romance",
    "thriller",
    "documentary",
    "animation",
    "other",
}


def clean_raw_output(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_and_validate_json(
    raw_text: str, prompt_version: str = "v2"
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    cleaned = clean_raw_output(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as err:
                return False, None, f"JSON decode failed: {str(err)}"
        else:
            return False, None, "No JSON object structure found."

    if not isinstance(data, dict):
        return False, None, "Parsed JSON is not an object/dictionary."

    required_keys = {"category", "sentiment", "entities"}
    if not required_keys.issubset(data.keys()):
        return False, None, f"Missing required keys: {required_keys - set(data.keys())}"

    # 1. Sentiment Normalizasyonu & Kontrolü
    sentiment_val = str(data.get("sentiment", "")).strip().lower()

    # --- v1 Kodu (Arşiv) ---
    # if sentiment_val in {"mixed", "ambivalent", "uncertain"}:
    #     sentiment_val = "neutral"
    # if sentiment_val not in VALID_SENTIMENTS_V1:
    #     return False, None, f"Invalid sentiment '{sentiment_val}'."

    # --- v2 Doğrulaması (İkili Mod) ---
    valid_sentiments = VALID_SENTIMENTS_V2 if prompt_version == "v2" else {"positive", "negative", "neutral"}
    if sentiment_val not in valid_sentiments:
        return False, None, f"Invalid sentiment '{sentiment_val}'. Allowed: {valid_sentiments}"
    data["sentiment"] = sentiment_val

    # 2. Category Normalizasyonu & Kontrolü
    category_val = str(data.get("category", "")).strip().title()
    if category_val.lower() not in VALID_CATEGORIES:
        category_val = "Other"
    data["category"] = category_val

    # 3. Entities Kontrolü
    if not isinstance(data.get("entities"), list):
        return False, None, "'entities' field must be a list."

    return True, data, ""

# ==============================================================================
# STAGE-WISE PARSING + STRICT SCHEMA CHECK (used by pipeline_settings.py)
# These helpers do NOT change the behaviour of parse_and_validate_json above.
# ==============================================================================
STRICT_CATEGORIES = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi",
    "Romance", "Thriller", "Documentary", "Animation", "Other",
]
STRICT_SENTIMENTS = ["positive", "negative"]


def parse_json_stages(raw_text: Any) -> Dict[str, Any]:
    """Report parsing success at two separate stages.

    raw_ok     : json.loads(raw_text) succeeds directly, BEFORE any cleaning.
    cleaned_ok : after clean_raw_output() (strip ``` fences) and, if needed, extracting
                 the first {...} block (drops conversational text around the JSON).
    """
    if not isinstance(raw_text, str):
        return {"raw_ok": False, "cleaned_ok": False, "data": None, "error": "Output is not a string."}

    raw_ok = False
    try:
        json.loads(raw_text)
        raw_ok = True
    except json.JSONDecodeError:
        pass

    cleaned = clean_raw_output(raw_text)
    data, error = None, ""
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            error = "No JSON object structure found."
        else:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError as err:
                error = f"JSON decode failed: {err}"

    cleaned_ok = isinstance(data, dict)
    if data is not None and not cleaned_ok:
        error = "Parsed JSON is not an object/dictionary."
        data = None
    return {"raw_ok": raw_ok, "cleaned_ok": cleaned_ok, "data": data, "error": error}


def strict_schema_check(data: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """All required fields present with correct types AND allowed values.

    Unlike parse_and_validate_json, nothing is silently repaired
    (e.g. an unknown category is NOT coerced to "Other"; it is a violation).
    Extra keys are tolerated.
    """
    if not isinstance(data, dict):
        return False, "not an object"
    missing = {"category", "sentiment", "entities"} - set(data)
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if data["category"] not in STRICT_CATEGORIES:
        return False, f"category not allowed: {data['category']!r}"
    if data["sentiment"] not in STRICT_SENTIMENTS:
        return False, f"sentiment not allowed: {data['sentiment']!r}"
    ents = data["entities"]
    if not isinstance(ents, list) or not all(isinstance(e, str) for e in ents):
        return False, "entities must be a list of strings"
    return True, ""
