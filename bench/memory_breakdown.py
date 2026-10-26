"""
memory_breakdown.py - how much memory each part of the pipeline adds (RSS and private bytes, after each step).

Usage: uv run python bench/memory_breakdown.py <pdf> --page 7
"""
import argparse
import gc
import time

import psutil

P = psutil.Process()
last = [0]


def mark(step):
    gc.collect()
    m = P.memory_info()
    private = getattr(m, "private", m.rss)
    print(f"{step:45} rss {m.rss / 2**20:7.0f} MB   private {private / 2**20:7.0f} MB   "
          f"(+{(m.rss - last[0]) / 2**20:6.0f} MB rss)")
    last[0] = m.rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--page", type=int, default=1)
    a = ap.parse_args()
    mark("python start")
    import numpy  # noqa: F401
    mark("numpy")
    from docling.datamodel.base_models import InputFormat
    from pdf_reader.formula_onnx import make_converter
    mark("import docling + pipeline module")
    conv = make_converter(4)
    conv.initialize_pipeline(InputFormat.PDF)
    import sys
    mark(f"models loaded (layout, table, formula); torch imported: {'torch' in sys.modules}")
    t = time.perf_counter()
    doc = conv.convert(a.pdf, page_range=(a.page, a.page)).document
    mark(f"converted page {a.page} ({time.perf_counter() - t:.1f} s)")
    doc = conv.convert(a.pdf, page_range=(a.page, a.page)).document
    mark(f"converted page {a.page} again")
    print("tables:", len(doc.tables), " formulas:", sum(1 for t in doc.texts if str(t.label) == "formula"))
    from pdf_reader.read_aloud import Narrator
    Narrator()
    mark("piper voice loaded")


if __name__ == "__main__":
    main()
