"""Prompt templates and strict schema definitions for Project 1.
Supports Baseline (v1), Optimized Binary (v2), and Heuristic-Guided (v3) prompts.
"""

# ==============================================================================
# BASELINE PROMPT (v1) - [ARCHIVED]
# ==============================================================================
# ALLOWED_CATEGORIES_V1 = [ ... ]
# ALLOWED_SENTIMENTS_V1 = ["positive", "negative", "neutral"]

# ==============================================================================
# OPTIMIZED PROMPT (v2) - [ARCHIVED]
# ==============================================================================
# ALLOWED_SENTIMENTS_V2 = ["positive", "negative"]
# SYSTEM_PROMPT_V2 = """ ... """

# ==============================================================================
# HEURISTIC-GUIDED PROMPT (v3) - [ACTIVE / ERROR-ANALYSIS DRIVEN]
# ==============================================================================
ALLOWED_CATEGORIES = [
    "Action",
    "Comedy",
    "Drama",
    "Horror",
    "Sci-Fi",
    "Romance",
    "Thriller",
    "Documentary",
    "Animation",
    "Other",
]

ALLOWED_SENTIMENTS = ["positive", "negative"]

SYSTEM_PROMPT_V3 = f"""You are a specialized, high-precision NLP engine that extracts structured data from movie reviews.
You must return ONLY a single valid JSON object strictly matching this schema:

{{
  "category": "Action | Comedy | Drama | Horror | Sci-Fi | Romance | Thriller | Documentary | Animation | Other",
  "sentiment": "positive | negative",
  "entities": ["string"]
}}

CRITICAL CLASSIFICATION & EXTRACTION RULES:
1. 'sentiment' MUST be strictly one of: {ALLOWED_SENTIMENTS}.
   - POLARITY RESOLUTION & HEURISTICS:
     a) RATING & NUMERICAL CUES: If the reviewer mentions an explicit low rating (e.g., 'give it a 3/10', 'not given a 1', '1 star'), classify as 'negative'.
     b) SOLE REDEEMING FACTOR: If the author praises only ONE isolated aspect (e.g., superb acting by one actor, beautiful lead actress, good visuals) but criticizes the rest of the film as silly, unmemorable, or awful, the OVERALL sentiment is strictly 'negative'.
     c) BALANCED REVIEWS: In reviews presenting both strengths and flaws, give greater weight to the final concluding verdict rather than introductory polite remarks.
     d) Do NOT output 'neutral', 'mixed', or any other label.
2. 'category' MUST be strictly one of: {ALLOWED_CATEGORIES}. Assign the single most dominant genre. Do NOT output generic terms like 'movie' or 'film'.
3. 'entities' MUST strictly contain specific proper nouns: real actors, directors, characters, or movie titles explicitly named.
   - NOISE FILTER: Do NOT include descriptive phrases, review commentary (e.g., 'great acting', 'waste of time'), or general common nouns. If no specific named entities exist, provide [].
4. Return raw JSON only. Absolutely NO markdown backticks (no ``` or ```json) and NO conversational filler.
"""

# Geriye dönük uyumluluk ve varsayılan aktif prompt:
SYSTEM_PROMPT = SYSTEM_PROMPT_V3


def get_system_prompt(version: str = "v3") -> str:
    # return SYSTEM_PROMPT_V1 if version == "v1" else (SYSTEM_PROMPT_V2 if version == "v2" else SYSTEM_PROMPT_V3)
    return SYSTEM_PROMPT_V3


def build_user_prompt(review_text: str) -> str:
    return f"""Review text:
\"\"\"
{review_text.strip()}
\"\"\"

Output JSON:"""