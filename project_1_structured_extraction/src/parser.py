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