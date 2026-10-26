"""
measure_resources.py - wall time and peak memory (RSS) of one pipeline run, sampled every 0.2 s.

Usage (main project env):
    uv run python bench/measure_resources.py codeformula test_papers/p5.pdf
    uv run python bench/measure_resources.py onnx test_papers/a_probabilistic_numeric_method.pdf --pages 5-14

  codeformula  Docling with CodeFormulaV2 (light settings), the starting point - needs PyTorch:
               uv sync --group export
  onnx         this project's pipeline (PP-FormulaNet_plus-S on ONNX Runtime)
"""
import argparse
import threading
import time

import psutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["codeformula", "onnx"])
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-9999")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.pages.split("-"))

    proc, peak, done = psutil.Process(), [0], threading.Event()

    def sample():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.2)

    threading.Thread(target=sample, daemon=True).start()
    proc.cpu_percent()
    t0 = time.perf_counter()
    if a.mode == "onnx":
        from pdf_reader.formula_onnx import make_converter
        conv = make_converter(4)
    else:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import CodeFormulaVlmOptions, PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.table_structure_options.mode = TableFormerMode.FAST
        opts.do_formula_enrichment = True
        opts.code_formula_options = CodeFormulaVlmOptions.from_preset("codeformulav2", expansion_factor=0.08, scale=1.5)
        opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    res = conv.convert(a.pdf, page_range=(lo, hi))
    wall = time.perf_counter() - t0
    cpu = proc.cpu_percent() / psutil.cpu_count()
    done.set()
    n = sum(1 for it, _ in res.document.iterate_items() if it.label == "formula")
    print(f"{a.mode} {len(res.pages)} pages, {n} formulas: wall {wall:.0f}s (incl. model load), "
          f"peak RAM {peak[0] / 2**30:.2f} GB, avg CPU {cpu:.0f}% of all threads")


if __name__ == "__main__":
    main()
