"""
Report figures for the prompt-only vs schema comparison, built from the saved result CSVs
(no model run needed). Saves PNGs next to the existing plots.

Run from the repository root:
    python project_1_structured_extraction/src/make_plots.py
    python project_1_structured_extraction/src/make_plots.py --temperature 1.0

Figures (reports/plots/):
    10_format_robustness.png    raw json.loads / after cleaning / schema-valid, per prompt style
    11_entity_f1.png            entity F1 (strict and lenient) for all four conditions
    12_entity_error_types.png   where the entity errors come from (schema, hardened prompt)

Colors: blue = prompt-only, orange = schema. This pair passes the CVD / contrast checks
of the dataviz palette validator (worst adjacent CVD DeltaE 24.7). Every bar is also
labelled with its value, so identity never depends on color alone.
Prompt-only entity F1 uses the "salvaged" variant (entities of schema-invalid rows are
kept when the JSON parses), which is the fair comparison.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from project_1_structured_extraction.src.evaluate_settings import (  # noqa: E402
    GOLD_CSV,
    RESULT_DIR,
    evaluate_one,
    lenient_match,
    load_gold,
    norm_entity,
    salvaged_entities,
)

PLOT_DIR = "project_1_structured_extraction/reports/plots"
SEED = 42

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"    # prompt-only (categorical slot 1)
ORANGE = "#eb6834"  # schema      (categorical slot 2)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK,
})


def load(style, temp, setting):
    suffix = "" if style == "hardened" else f"_{style}"
    path = os.path.join(RESULT_DIR, f"{setting}{suffix}_T{temp:.1f}_seed{SEED}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["entities_json"] = df["entities_json"].fillna("[]")
    df["done_reason"] = df["done_reason"].fillna("")
    return df


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(ax, groups, a_vals, b_vals, fmt):
    x = np.arange(len(groups))
    w = 0.36
    kw = dict(edgecolor=SURFACE, linewidth=1.5)
    ba = ax.bar(x - w / 2, a_vals, w, color=BLUE, label="prompt-only", **kw)
    bb = ax.bar(x + w / 2, b_vals, w, color=ORANGE, label="schema (format=JSON schema)", **kw)
    for bars in (ba, bb):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01 * ax.get_ylim()[1],
                    fmt(r.get_height()), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    style_axes(ax)


def fig_format(temp, out):
    styles = [s for s in ("hardened", "minimal") if load(s, temp, "prompt_only") is not None]
    if not styles:
        print("[skip] no results for figure 10")
        return
    fig, axes = plt.subplots(1, len(styles), figsize=(7.4, 3.3), sharey=True, squeeze=False)
    names = ["raw json.loads", "after cleaning", "schema-valid"]
    for ax, st in zip(axes[0], styles):
        po = evaluate_one(load(st, temp, "prompt_only"), None)
        sc = evaluate_one(load(st, temp, "schema"), None)
        keys = ["raw_json_rate", "cleaned_json_rate", "schema_valid_rate"]
        ax.set_ylim(0, 108)
        grouped_bars(ax, names, [po[k] * 100 for k in keys], [sc[k] * 100 for k in keys], lambda v: f"{v:.1f}%")
        ax.set_title(f"{st.capitalize()} prompt, T={temp}")
        ax.set_yticks([0, 25, 50, 75, 100])
    axes[0][0].set_ylabel("% of 200 outputs")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def fig_entity(gold, out):
    conds = [(s, t) for s in ("hardened", "minimal") for t in (0.0, 1.0)
             if load(s, t, "prompt_only") is not None and load(s, t, "schema") is not None]
    if not conds or not gold:
        print("[skip] no results / gold for figure 11")
        return
    labels = [f"{s}\nT={t}" for s, t in conds]
    vals = {k: {"po": [], "sc": []} for k in ("strict", "lenient")}
    for s, t in conds:
        po = evaluate_one(load(s, t, "prompt_only"), gold)
        sc = evaluate_one(load(s, t, "schema"), gold)
        for k in vals:
            vals[k]["po"].append(po[f"ent_{k}_F1_salvaged"])
            vals[k]["sc"].append(sc[f"ent_{k}_F1_salvaged"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, k in zip(axes, ("strict", "lenient")):
        ax.set_ylim(0, 1.0)
        grouped_bars(ax, labels, vals[k]["po"], vals[k]["sc"], lambda v: f"{v:.2f}")
        ax.set_title(f"Entity F1, {k} match")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("micro F1 (100 annotated reviews)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def error_counts(df, gold):
    c = defaultdict(int)
    for _, r in df[df["id"].isin(gold)].iterrows():
        pred = {norm_entity(e) for e in salvaged_entities(r["raw_output"]) if norm_entity(e)}
        gmap = {norm_entity(e) for e in gold[int(r["id"])] if norm_entity(e)}
        for p in pred - gmap:
            c["fp_boundary" if any(lenient_match(p, g) for g in gmap) else "fp_pure"] += 1
        for g in gmap - pred:
            c["fn_boundary" if any(lenient_match(g, p) for p in pred) else "fn_pure"] += 1
    return c


def fig_errors(gold, temp, out):
    df = load("hardened", temp, "schema")
    if df is None or not gold:
        print("[skip] no results / gold for figure 12")
        return
    c = error_counts(df, gold)
    rows = [
        ("False positive, pure\n(extra entity not in gold)", c["fp_pure"]),
        ("False negative, pure\n(gold entity never produced)", c["fn_pure"]),
        ("False positive, boundary\n(partial match, e.g. '(2005)' added)", c["fp_boundary"]),
        ("False negative, boundary\n(partial match to a prediction)", c["fn_boundary"]),
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    y = np.arange(len(rows))
    bars = ax.barh(y, [r[1] for r in rows], height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    for b in bars:
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("number of entity errors (strict match)")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.12 + 2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Entity errors, schema setting, hardened prompt, T={temp}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature for figures 10 and 12.")
    args = ap.parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)
    gold = load_gold(GOLD_CSV)
    fig_format(args.temperature, os.path.join(PLOT_DIR, "10_format_robustness.png"))
    fig_entity(gold, os.path.join(PLOT_DIR, "11_entity_f1.png"))
    fig_errors(gold, args.temperature, os.path.join(PLOT_DIR, "12_entity_error_types.png"))


if __name__ == "__main__":
    main()
