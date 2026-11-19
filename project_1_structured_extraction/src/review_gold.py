"""
Interactive terminal tool to VERIFY / correct the gold entity annotations, one review
per screen: the review text with the gold entities marked as [[Entity]], the numbered
gold list, and capitalized candidates that are NOT in the gold list (to spot misses).

Run from the repository root:
    python project_1_structured_extraction/src/review_gold.py

Commands at the prompt:
    <Enter>         accept this row as is (marks it verified)
    d 2 4           delete gold entities number 2 and 4
    a Name; Name2   add entities (separate with ;)
    r A; B; C       replace the whole list
    n               no entities (writes NONE)
    s               skip for now (not marked verified)
    q               save and quit

Progress is saved after every row; rerun the script to resume where you stopped.
Added columns: `verified` (yes once you accepted/edited the row) and `changed`
(yes if you modified the original annotation) -> use these counts in the report.

Annotation guideline reminder: named persons (actors, directors, characters) and
titles of films/shows/books. NOT genres, studios, teams, places, roles, pronouns.
If one entity appears in several forms, keep only the longest form.
"""

import argparse
import re
import sys
import textwrap

import pandas as pd

GOLD_CSV = "project_1_structured_extraction/data/annotation/entity_gold.csv"


def parse_gold(cell):
    s = "" if pd.isna(cell) else str(cell).strip()
    return [] if s == "" or s.upper() == "NONE" else [x.strip() for x in s.split(";") if x.strip()]


def to_cell(entities):
    return "; ".join(entities) if entities else "NONE"


def mark(text, entities):
    """Wrap every gold entity occurrence in [[ ]] (longest first, case-insensitive)."""
    if not entities:
        return text
    pattern = "|".join(re.escape(e) for e in sorted(entities, key=len, reverse=True))
    return re.sub(f"({pattern})", r"[[\1]]", text, flags=re.IGNORECASE)


def missing_candidates(cands, entities):
    low = [e.lower() for e in entities]
    out = []
    for c in [x.strip() for x in str(cands).split(";") if x.strip()] if pd.notna(cands) else []:
        if not any(c.lower() in e or e in c.lower() for e in low):
            out.append(c)
    return out


def show(row, entities, pos, total):
    print("\n" + "=" * 100)
    print(f"[{pos}/{total}]  id={int(row['id'])}  label={row['ground_truth_sentiment']}")
    print("=" * 100)
    print(textwrap.fill(mark(str(row["text"]), entities), width=100))
    print("-" * 100)
    print("GOLD:", "(none)" if not entities else "")
    for i, e in enumerate(entities, 1):
        print(f"  {i}. {e}")
    miss = missing_candidates(row.get("candidate_capitalized"), entities)
    print("Capitalized words NOT in gold:", "; ".join(miss) if miss else "-")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=GOLD_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.file, encoding="utf-8-sig")
    for col in ("verified", "changed"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    todo = [i for i in df.index if df.at[i, "verified"] != "yes"]
    print(f"{len(df)} rows, {len(df) - len(todo)} already verified, {len(todo)} to review.")
    quit_now = False
    for n, i in enumerate(todo, 1):
        row = df.loc[i]
        original = parse_gold(row["gold_entities"])
        ents = list(original)
        while True:
            show(row, ents, len(df) - len(todo) + n, len(df))
            cmd = input("[Enter]=ok  d N..  a X;Y  r X;Y  n  s  q > ").strip()
            low = cmd.lower()
            if cmd == "":
                break
            if low == "q":
                quit_now = True
                break
            if low == "s":
                ents = None
                break
            if low == "n":
                ents = []
            elif low.startswith("d "):
                idx = {int(x) for x in cmd[2:].split() if x.isdigit()}
                ents = [e for k, e in enumerate(ents, 1) if k not in idx]
            elif low.startswith("a "):
                ents += [x.strip() for x in cmd[2:].split(";") if x.strip()]
            elif low.startswith("r "):
                ents = [x.strip() for x in cmd[2:].split(";") if x.strip()]
            else:
                print("Unknown command.")
        if quit_now:
            break
        if ents is None:  # skipped
            continue
        df.at[i, "gold_entities"] = to_cell(ents)
        df.at[i, "verified"] = "yes"
        df.at[i, "changed"] = "yes" if ents != original else "no"
        df.to_csv(args.file, index=False, encoding="utf-8-sig")  # autosave

    df.to_csv(args.file, index=False, encoding="utf-8-sig")
    v = (df["verified"] == "yes").sum()
    c = (df["changed"] == "yes").sum()
    print(f"\nSaved. Verified {v}/{len(df)} rows; changed {c} of them.")


if __name__ == "__main__":
    main()
