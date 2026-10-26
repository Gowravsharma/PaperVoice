"""
eval_layout.py - compare PaperVoice runs that used different layout models, against the arXiv source.

Usage (main project env, needs docling_core):
    uv run python bench/eval_layout.py refs/arxiv_0905.1863/FahimTouziWarin-final.tex \
        heron=out/layout_heron/a_probabilistic_numeric_method.json \
        egret_m=out/layout_egret_m/a_probabilistic_numeric_method.json

Each run's formula items are aligned to the source rows with the same text-layer alignment as build_truth.py
(no recognizer output is used for matching), then the recognised LaTeX is scored against the matched rows.
Reports: formulas detected, matched to a real equation, source rows covered, rows only one model found,
and recognition quality.
Historical (section 9): compares layout outputs saved earlier.
"""
import sys
from pathlib import Path
from statistics import mean

from docling_core.types.doc import DoclingDocument

sys.path.insert(0, str(Path(__file__).parent))
from build_truth import align, atoms_pdf, sim, source_rows  # noqa: E402
from common import score  # noqa: E402


def evaluate(json_path, rows):
    doc = DoclingDocument.load_from_json(Path(json_path))
    items = [it for it, _ in doc.iterate_items() if it.label == "formula"]
    crops = [{"atoms": atoms_pdf(it.orig)} for it in items]
    match = align(crops, rows)
    covered, scores, weak = set(), [], 0
    for i, it in enumerate(items):
        idx = match.get(i)
        if not idx:
            continue
        joined = [a for r in idx for a in rows[r]["atoms"]]
        weak += sim(crops[i]["atoms"], joined) < 0.8
        covered.update(idx)
        scores.append(score(it.text, " \\\\ ".join(rows[r]["latex"] for r in idx)))
    return {"detected": len(items), "matched": len(scores), "weak": weak,
            "covered": covered, "scores": scores}


def main():
    rows = source_rows(sys.argv[1])
    runs = dict(arg.split("=", 1) for arg in sys.argv[2:])
    res = {name: evaluate(path, rows) for name, path in runs.items()}

    print(f"source: {len(rows)} display-math rows\n")
    print(f"{'layout':<10} {'detected':>8} {'matched':>8} {'unmatched':>9} {'rows covered':>12} {'quality':>8} {'<0.80':>6}")
    for name, r in res.items():
        s = r["scores"]
        print(f"{name:<10} {r['detected']:>8} {r['matched']:>8} {r['detected'] - r['matched']:>9} "
              f"{len(r['covered']):>12} {mean(s):>8.3f} {sum(x < 0.8 for x in s) / len(s):>6.1%}")
    if len(res) == 2:
        (a, ra), (b, rb) = res.items()
        for x, rx, y, ry in ((a, ra, b, rb), (b, rb, a, ra)):
            only = sorted(rx["covered"] - ry["covered"])
            print(f"\nrows only {x} found ({len(only)}):")
            for r in only[:10]:
                print(f"  {rows[r]['latex'][:110]}")


if __name__ == "__main__":
    main()
