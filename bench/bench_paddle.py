"""
bench_paddle.py - time PaddleOCR formula recognizers on CPU, one crop at a time.

Usage (bench venv, Python 3.12):
    .venv/Scripts/python bench_paddle.py [--models PP-FormulaNet-S PP-FormulaNet_plus-S] [--threads 4]
"""
import argparse
import json
import time

from paddleocr import FormulaRecognition

from common import ROOT, load_crops, load_truth, score


def bench(model_name, crops, truth, threads):
    t0 = time.perf_counter()
    model = FormulaRecognition(model_name=model_name, device="cpu", cpu_threads=threads)
    load = time.perf_counter() - t0
    model.predict(str(crops[0]))  # warm-up

    rows = []
    for crop, ref in zip(crops, truth):
        t0 = time.perf_counter()
        latex = model.predict(str(crop))[0]["rec_formula"]
        rows.append({"file": crop.name, "sec": time.perf_counter() - t0,
                     "score": score(latex, ref), "latex": latex})
    return load, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["PP-FormulaNet-S", "PP-FormulaNet_plus-S"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    crops, truth = load_crops(), load_truth()

    results = {}
    for name in a.models:
        load, rows = bench(name, crops, truth, a.threads)
        results[name] = {"load": load, "rows": rows}
        print(f"\n=== {name} ===  load {load:.1f}s")
        for r in rows:
            print(f"  {r['file']}  {r['sec']:6.2f}s  score {r['score']:.3f}  {r['latex']}")
        n = len(rows)
        print(f"  mean {sum(r['sec'] for r in rows) / n:.2f}s/formula  "
              f"mean score {sum(r['score'] for r in rows) / n:.3f}")

    out = ROOT / "out" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "paddle.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
