"""Aggregate Clean-set VQA results from vlm_results/ into a camera-ready
LaTeX table (Base / Ours / $\\Delta$, no typographic attack).

Usage:
    python vlm_results_table.py [--results-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import os

RESULTS_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "vlm_results")
CLEAN_COND = "mc_clean"

# (filename stem, display label). InternVL rows use the uppercase-HF variant
# only. Qwen rows use the _prompt variant — mc_clean only ships the _prompt
# jsonl for Qwen models.
MODELS = [
    ("google__gemma-3-4b-it",                   "Gemma3-4B"),
    ("google__gemma-3-12b-it",                  "Gemma3-12B"),
    ("OpenGVLab__InternVL3_5-8B-HF",            "InternVL3.5-8B"),
    ("OpenGVLab__InternVL3_5-14B-HF",           "InternVL3.5-14B"),
    ("Qwen__Qwen3-VL-4B-Instruct_prompt",       "Qwen3-VL-4B"),
    ("Qwen__Qwen3-VL-8B-Instruct_prompt",       "Qwen3-VL-8B"),
    ("Qwen__Qwen3-VL-30B-A3B-Instruct_prompt",  "Qwen3-VL-30B-A3B"),
]


def load_records(path):
    """Load prediction records from a results JSONL, skipping spec headers
    and de-duplicating by qid (keeping the last occurrence)."""
    by_qid = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "qid" not in obj:
                continue
            by_qid[obj["qid"]] = obj
    return list(by_qid.values())


def score_clean(path):
    """Return (base_acc, ours_acc) on the clean split, in [0, 1]."""
    records = load_records(path)
    n = len(records)
    if n == 0:
        return None, None
    base_correct = 0
    ours_correct = 0
    for r in records:
        gt = r.get("answer", "")
        if (r.get("base_pred_answer") or "").strip() == gt:
            base_correct += 1
        if (r.get("pred_answer") or "").strip() == gt:
            ours_correct += 1
    return base_correct / n, ours_correct / n


def collect(results_dir):
    """Return [(label, base_acc, ours_acc)] in MODELS order; missing → None."""
    out = []
    for stem, label in MODELS:
        path = os.path.join(results_dir, CLEAN_COND, f"{stem}.jsonl")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}")
            out.append((label, None, None))
            continue
        b, o = score_clean(path)
        out.append((label, b, o))
    return out


def fmt_pct(x):
    return f"{100 * x:.1f}"


def fmt_row(label, base, ours):
    if base is None or ours is None:
        return f"{label} & -- & -- & -- \\\\"
    bs, os_ = fmt_pct(base), fmt_pct(ours)
    if ours > base:
        bs_cell, os_cell = bs, rf"\textbf{{{os_}}}"
    elif base > ours:
        bs_cell, os_cell = rf"\textbf{{{bs}}}", os_
    else:
        bs_cell, os_cell = bs, os_
    d = 100 * (ours - base)
    d_cell = f"{d:+.1f}"
    if d > 0:
        d_cell = rf"\textbf{{{d_cell}}}"
    return f"{label} & {bs_cell} & {os_cell} & {d_cell} \\\\"


def build_clean_table(rows, caption, label):
    body_lines = [fmt_row(lbl, b, o) for lbl, b, o in rows]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Base & Ours & $\Delta$ \\",
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=os.path.dirname(__file__))
    args = ap.parse_args()

    rows = collect(args.results_dir)

    print(f"\n=== Clean VQA accuracy ({CLEAN_COND}) ===")
    print(f"{'model':20s} {'base':>7s} {'ours':>7s} {'Δ':>7s}")
    for label, b, o in rows:
        if b is None:
            print(f"{label:20s} {'--':>7s} {'--':>7s} {'--':>7s}")
            continue
        print(f"{label:20s} {100*b:7.2f} {100*o:7.2f} {100*(o-b):+7.2f}")

    tex = build_clean_table(
        rows,
        caption=("Clean VQA accuracy (no typographic attack) with/without "
                 "intervention. $\\Delta$ shows the change."),
        label="tab:clean_vqa",
    )
    print("\n% ===== Clean VQA accuracy table =====")
    print(tex)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "vlm_vqa_clean.tex")
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
