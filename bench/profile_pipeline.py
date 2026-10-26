"""
profile_pipeline.py - time each Docling stage on CPU for several pipeline configs.

Usage:
    uv run python bench/profile_pipeline.py test_papers/p5.pdf [--pages 1-1] [--configs default light light_egret no_formula]

Model load (initialize_pipeline) is timed separately from conversion.
Writes per-config Markdown to out/profile/<config>.md for a quality check.
Historical (sections 1 and 9): the Docling PyTorch configs need uv sync --group export.
"""
import argparse
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

settings.debug.profile_pipeline_timings = True


def default_opts():
    opts = PdfPipelineOptions()
    opts.do_formula_enrichment = True
    opts.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
    return opts


def light_opts(layout="layout_heron_default", formula=True):
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = formula
    opts.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2", expansion_factor=0.08, scale=1.5
    )
    opts.layout_options = LayoutObjectDetectionOptions.from_preset(layout)
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    return opts


CONFIGS = {
    "default": default_opts,
    "light": light_opts,
    "light_egret": lambda: light_opts(layout="layout_egret_medium"),
    "no_formula": lambda: light_opts(formula=False),
}


def run(name, pdf, page_range):
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=CONFIGS[name]())}
    )
    t0 = time.perf_counter()
    conv.initialize_pipeline(InputFormat.PDF)
    load = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = conv.convert(pdf, page_range=page_range)
    total = time.perf_counter() - t0

    out = Path("out/profile")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.md").write_text(res.document.export_to_markdown(), encoding="utf-8")
    n_formulas = sum(1 for it, _ in res.document.iterate_items() if it.label == "formula")

    print(f"\n=== {name} ===  load {load:.1f}s  convert {total:.1f}s  formulas {n_formulas}")
    for key, item in sorted(res.timings.items(), key=lambda kv: -kv[1].total()):
        if item.total() >= 0.05:
            print(f"  {key:<32} {item.total():7.2f}s  (x{item.count})")
    return load, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-1")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.pages.split("-"))

    summary = {name: run(name, a.pdf, (lo, hi)) for name in a.configs}
    print("\nconfig          load(s)  convert(s)")
    for name, (load, total) in summary.items():
        print(f"  {name:<14} {load:7.1f}  {total:9.1f}")


if __name__ == "__main__":
    main()
