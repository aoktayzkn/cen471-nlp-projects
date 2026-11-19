"""
Builds the manual entity-annotation file (gold entities) for a balanced subset.

Why this is needed: IMDB only has sentiment labels, so entity-extraction F1 needs
gold entities annotated by hand (assignment: "you must manually annotate the
entities in your evaluation samples and describe your annotation rules").

Run from the repository root:
    python project_1_structured_extraction/src/build_annotation_template.py

Then open data/annotation/entity_gold.csv in Excel / LibreOffice / VS Code and fill
the `gold_entities` column. The `candidate_capitalized` column is only a reading aid
(regex over capitalized words; it does NOT come from the LLM, so it cannot bias
the evaluation). Do not trust it blindly.

Annotation format for `gold_entities`:
    - separate entities with a semicolon:  Gene Roddenberry; Babylon 5; Star Trek
    - if the review names NO entity, write exactly:  NONE
    - leave the cell empty only for rows you have not annotated yet
"""

import os
import re

import pandas as pd

INPUT_CSV = "project_1_structured_extraction/data/raw/imdb_balanced_200.csv"
OUTPUT_CSV = "project_1_structured_extraction/data/annotation/entity_gold.csv"
PER_CLASS = 50  # 50 + 50 = 100 samples (assignment minimum is 100)
SEED = 42

STOPWORDS = {
    "I", "A", "An", "The", "This", "That", "These", "Those", "It", "Its", "He", "She", "They",
    "We", "You", "My", "Our", "His", "Her", "In", "On", "At", "Of", "For", "And", "But", "Or",
    "So", "If", "As", "When", "While", "What", "Why", "How", "Who", "There", "Here", "Then",
    "Not", "No", "Yes", "Well", "Oh", "Also", "However", "Although", "Maybe", "Overall",
    "Br", "Is", "Was", "Be", "Do", "Did", "Just", "Even", "After", "Before", "One", "Some",
}
CAP_SEQ = re.compile(r"[A-Z][A-Za-z'’\-]*(?:[ \t]+(?:of|the|de|van|von|la|le)?[ \t]*[A-Z][A-Za-z'’\-]*)*")
CONTRACTION = re.compile(r"^(I|It|That|He|She|They|We|There|What|Here|Let|You|Who|Don|Didn|Doesn|Isn|Wasn)['’]", re.I)


def candidates(text):
    found = []
    for m in CAP_SEQ.finditer(text.replace("<br />", " ")):
        s = m.group(0).strip(" .'’-")
        if s and s not in STOPWORDS and len(s) > 1 and not CONTRACTION.match(s) and s not in found:
            found.append(s)
    return "; ".join(found)


def main():
    df = pd.read_csv(INPUT_CSV)
    parts = [
        g.sample(n=PER_CLASS, random_state=SEED)
        for _, g in df.groupby("ground_truth_sentiment")
    ]
    sub = pd.concat(parts).sort_values("id").reset_index(drop=True)
    sub["text"] = sub["text"].str.replace("<br />", " ", regex=False)
    sub["candidate_capitalized"] = sub["text"].apply(candidates)
    sub["gold_entities"] = ""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    if os.path.exists(OUTPUT_CSV):
        raise SystemExit(f"{OUTPUT_CSV} already exists; refusing to overwrite your annotations.")
    sub[["id", "ground_truth_sentiment", "text", "candidate_capitalized", "gold_entities"]].to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote {len(sub)} rows to {OUTPUT_CSV} ({PER_CLASS} per class, seed={SEED}).")


if __name__ == "__main__":
    main()
