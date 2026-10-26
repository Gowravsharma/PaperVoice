"""
memory_over_pages.py - memory of the reading engine while it indexes a document page after page, as the server does.

Usage: uv run python bench/memory_over_pages.py <pdf> [--pages 30]
Prints resident and committed (private) memory after each page; a steady climb means something is kept per page.
"""
import argparse
import gc
import time

import psutil

from pdf_reader.stream import Engine
from pdf_reader.text_norm import FAILURES  # noqa: F401  (same imports as the server)


def mem():
    m = psutil.Process().memory_info()
    return m.rss / 2**20, getattr(m, "private", m.rss) / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, default=30)
    a = ap.parse_args()
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(a.pdf)
    n = min(a.pages, len(pdf))
    eng = Engine(4)
    print(f"engine loaded: rss {mem()[0]:.0f} MB, private {mem()[1]:.0f} MB")
    state, stats = {"keep_references": False, "in_refs": False}, {}
    from pdf_reader.script import new_stats
    for p in range(1, n + 1):
        t = time.perf_counter()
        blocks = eng.index_page(a.pdf, pdf, p, False, state, new_stats())
        gc.collect()
        r, pr = mem()
        print(f"page {p:3d}: {len(blocks):3d} blocks in {time.perf_counter() - t:4.1f} s   rss {r:6.0f} MB   private {pr:6.0f} MB",
              flush=True)
    eng.close()


if __name__ == "__main__":
    main()
