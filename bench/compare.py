"""
compare.py - CodeFormulaV2 vs PP-FormulaNet_plus-S (ONNX) on the same crops of the full paper.

Inputs:
    refs/fahim_ground_truth.json      truth per crop (build_truth.py)
    crops/fahim_cf/{manifest,codeformula}.json   CodeFormula LaTeX for a page subset (bench/extract_crops.py --codeformula)
    out/bench/onnx_fahim.json         ONNX results for all crops (bench_onnx.py --set fahim)
CodeFormula crops are matched to the full crop set by page + bbox, so both models are scored on identical formulas.
"""
import json
from statistics import mean, median

from common import ROOT, score, tokens


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def summary(name, scores, secs=None):
    n = len(scores)
    line = (f"  {name:<22} n={n:<4} mean {mean(scores):.3f}  median {median(scores):.3f}  "
            f">=0.95: {sum(s >= 0.95 for s in scores) / n:5.1%}  <0.80: {sum(s < 0.8 for s in scores) / n:5.1%}")
    if secs:
        line += f"  {mean(secs):5.2f}s/formula"
    print(line)


def main():
    truth = {g["file"]: g for g in load("refs/fahim_ground_truth.json") if g["truth"]}
    full = {(m["page"], tuple(m["bbox"])): m["file"] for m in load("crops/fahim/manifest.json")}
    onnx = {r["file"]: r for r in load("out/bench/onnx_fahim.json")["onnx_fp32"]["rows"]}

    cf_man, cf = load("crops/fahim_cf/manifest.json"), load("crops/fahim_cf/codeformula.json")
    cf_by_file, unmatched = {}, 0
    for m, latex in zip(cf_man, cf["latex"]):
        f = full.get((m["page"], tuple(m["bbox"])))
        if f is None:
            unmatched += 1
        elif f in truth:
            cf_by_file[f] = latex
    per_cf = cf["formula_stage_sec"] / len(cf["latex"])

    print(f"CodeFormula subset: {len(cf_man)} crops, {len(cf_by_file)} with truth, {unmatched} boxes not in full set")
    print(f"CodeFormula formula stage {cf['formula_stage_sec']:.0f}s -> {per_cf:.1f}s/formula (batched)\n")

    sub = sorted(cf_by_file)
    print("Same subset (pages 5-14):")
    summary("CodeFormulaV2", [score(cf_by_file[f], truth[f]["truth"]) for f in sub], [per_cf] * len(sub))
    summary("PP-FormulaNet+S ONNX", [onnx[f]["score"] for f in sub], [onnx[f]["sec"] for f in sub])

    files = sorted(f for f in truth if f in onnx)
    print("\nPP-FormulaNet+S ONNX, all formulas, by truth length (tokens):")
    buckets = [(0, 30), (30, 60), (60, 100), (100, 10**6)]
    for lo, hi in buckets:
        fs = [f for f in files if lo <= len(tokens(truth[f]["truth"])) < hi]
        if fs:
            summary(f"{lo}-{hi if hi < 10**6 else ''} tokens", [onnx[f]["score"] for f in fs], [onnx[f]["sec"] for f in fs])
    summary("all", [onnx[f]["score"] for f in files], [onnx[f]["sec"] for f in files])

    print("\nWorst ONNX results:")
    for f in sorted(files, key=lambda f: onnx[f]["score"])[:8]:
        print(f"  {f} p{truth[f]['page']} {onnx[f]['score']:.2f}\n     truth {truth[f]['truth'][:110]}\n     onnx  {onnx[f]['latex'][:110]}")
    if sub:
        print("\nWorst CodeFormula results on the subset:")
        for f in sorted(sub, key=lambda f: score(cf_by_file[f], truth[f]["truth"]))[:5]:
            print(f"  {f} {score(cf_by_file[f], truth[f]['truth']):.2f}\n     truth {truth[f]['truth'][:110]}\n     cf    {cf_by_file[f][:110]}")


if __name__ == "__main__":
    main()
