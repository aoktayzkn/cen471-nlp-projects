"""Prompt templates and strict schema definitions for Project 1."""

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

ALLOWED_SENTIMENTS = ["positive", "negative", "neutral"]

SYSTEM_PROMPT = f"""You are a specialized NLP engine that extracts structured data from movie reviews.
You must return ONLY a single valid JSON object strictly matching this schema:

{{
  "category": "Action | Comedy | Drama | Horror | Sci-Fi | Romance | Thriller | Documentary | Animation | Other",
  "sentiment": "positive | negative | neutral",
  "entities": ["string"]
}}

CRITICAL RULES:
1. 'category' MUST be strictly one of: {ALLOWED_CATEGORIES}. Do NOT output generic terms like 'movie' or 'film'.
2. 'sentiment' MUST be strictly one of: {ALLOWED_SENTIMENTS}. Do NOT invent new labels like 'mixed'.
3. 'entities' must be a list of actor, director, character names, or movie titles mentioned. If none, provide [].
4. Return raw JSON only. Absolutely NO markdown backticks (no ``` or ```json) and NO conversational filler.
"""


def build_user_prompt(review_text: str) -> str:
  return f"""Review text:
\"\"\"
{review_text.strip()}
\"\"\"

Output JSON:"""