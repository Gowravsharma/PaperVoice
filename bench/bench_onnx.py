"""
bench_onnx.py - run PP-FormulaNet_plus-S through ONNX Runtime on CPU, one crop at a time.

Preprocessing and token decoding are PaddleX's own; only the network call is swapped for ONNX Runtime,
so results are directly comparable with bench_paddle.py.

Usage (bench venv):
    .venv/Scripts/python bench_onnx.py [--set p5|fahim] [--threads 4] [--int8 [--quantize]]
      --int8       also run models/ppfn_plus_s.int8.onnx (dynamic int8; slower and worse on p5)
      --quantize   (re)build the int8 model first
Writes out/bench/onnx_<set>.json.
"""
import argparse
import json
import time
from pathlib import Path

import onnxruntime as ort
from paddleocr import FormulaRecognition

from common import ROOT, load_set, score

MODELS = ROOT / "models"
FP32 = MODELS / "ppfn_plus_s.onnx"
INT8 = MODELS / "ppfn_plus_s.int8.onnx"


class OrtRunner:
    """Stands in for PaddleX's inference runner: accepts runner([img]) or runner(x=batch)."""

    def __init__(self, path, threads):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def __call__(self, imgs=None, x=None):
        batch = x[0] if x is not None else imgs[0][None]
        return self.sess.run(None, {self.input: batch})


def quantize():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8)
    print(f"wrote {INT8} ({INT8.stat().st_size / 2**20:.0f} MB, fp32 {FP32.stat().st_size / 2**20:.0f} MB)")


def bench(path, model, items, threads, verbose):
    t0 = time.perf_counter()
    model.paddlex_predictor.runner = OrtRunner(path, threads)
    load = time.perf_counter() - t0
    model.predict(str(items[0][0]))  # warm-up
    rows = []
    for crop, ref in items:
        t0 = time.perf_counter()
        latex = model.predict(str(crop))[0]["rec_formula"]
        rows.append({"file": crop.name, "sec": time.perf_counter() - t0,
                     "score": score(latex, ref), "latex": latex})
        if verbose:
            r = rows[-1]
            print(f"  {r['file']}  {r['sec']:6.2f}s  score {r['score']:.3f}  {r['latex'][:100]}")
    return load, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="p5")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    a = ap.parse_args()
    if a.int8 and (a.quantize or not INT8.exists()):
        quantize()

    items = load_set(a.set)
    model = FormulaRecognition(model_name="PP-FormulaNet_plus-S", device="cpu")
    runs = [("onnx_fp32", FP32)] + ([("onnx_int8", INT8)] if a.int8 else [])
    results = {}
    for name, path in runs:
        print(f"\n=== {name} on {a.set} ({len(items)} formulas) ===")
        load, rows = bench(path, model, items, a.threads, verbose=len(items) <= 20)
        results[name] = {"load": load, "rows": rows}
        n = len(rows)
        print(f"  session load {load:.1f}s  mean {sum(r['sec'] for r in rows) / n:.2f}s/formula  "
              f"total {sum(r['sec'] for r in rows):.0f}s  mean score {sum(r['score'] for r in rows) / n:.3f}")

    out = ROOT / "out" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"onnx_{a.set}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
