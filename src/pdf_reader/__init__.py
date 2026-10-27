from pdf_reader import vc_runtime

vc_runtime.preload()  # first of all: a private C++ runtime for computers whose System32 one is too old (vc_runtime.py)

import argparse
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF -> Markdown/JSON with LaTeX formulas (CPU, ONNX formula model)")
    ap.add_argument("pdf")
    ap.add_argument("--pages", help="page range, e.g. 3-5")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--speech", action="store_true", help="also write <name>.speech.txt, a plain-text script for TTS")
    a = ap.parse_args()

    from docling.datamodel.settings import settings
    from pdf_reader.formula_onnx import make_converter

    settings.debug.profile_pipeline_timings = True
    t0 = time.perf_counter()
    page_range = tuple(int(x) for x in a.pages.split("-")) if a.pages else (1, 10**9)
    from pdf_reader.pdfium_safe import char_count, open_document, page_count
    pdf = open_document(a.pdf)
    no_text = [p for p in range(page_range[0], min(page_range[1], page_count(pdf)) + 1) if char_count(pdf, p) < 20]
    if no_text:  # scans / "Print to PDF" outlines: without OCR their paragraphs come out empty
        print(f"pages without a text layer: {no_text} -> OCR on")
    conv = make_converter(a.threads, ocr=bool(no_text))
    res = conv.convert(a.pdf, page_range=page_range)
    total = time.perf_counter() - t0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.pdf).stem
    res.document.save_as_markdown(out / f"{stem}.md")
    res.document.save_as_json(out / f"{stem}.json")
    n_formulas = sum(1 for it, _ in res.document.iterate_items() if it.label == "formula")
    stages = ", ".join(f"{k} {v.total():.1f}s" for k, v in sorted(res.timings.items(), key=lambda kv: -kv[1].total())
                       if k != "pipeline_total" and v.total() >= 0.5)
    print(f"{len(res.pages)} pages, {n_formulas} formulas in {total:.1f}s ({stages})")
    print(f"wrote {out / stem}.md and .json")
    if a.speech:
        from pdf_reader.script import write_script
        t0 = time.perf_counter()
        path, text, stats = write_script(out / f"{stem}.json", a.pdf)
        print(f"wrote {path} in {time.perf_counter() - t0:.1f}s: {len(text.split())} words; {stats}")
