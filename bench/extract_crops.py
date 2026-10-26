"""
extract_crops.py - save every layout box labelled `formula` as a PNG, without running a formula model.

Usage:
    uv run python bench/extract_crops.py test_papers/p5.pdf [--pages 1-1] [--scale 2.0] [--margin 0.08] [--out crops/p5]

Writes <out>/NN.png plus <out>/manifest.json (page, bbox, raw PDF text per crop).
`margin` is a fraction of the box height added on each side, like CodeFormula's expansion_factor.

  --codeformula  also run CodeFormulaV2 (light settings: expansion 0.08, scale 1.5) in the same pass and
                 store its LaTeX per crop plus the formula-stage time in <out>/codeformula.json.
"""
import argparse
import json
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import CodeFormulaVlmOptions, PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="1-9999")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--out", default=None)
    ap.add_argument("--codeformula", action="store_true")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.pages.split("-"))
    out = Path(a.out or f"crops/{Path(a.pdf).stem}")
    out.mkdir(parents=True, exist_ok=True)

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    opts.do_formula_enrichment = a.codeformula
    opts.code_formula_options = CodeFormulaVlmOptions.from_preset(
        "codeformulav2", expansion_factor=0.08, scale=1.5
    )
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    opts.generate_page_images = True
    opts.images_scale = a.scale
    settings.debug.profile_pipeline_timings = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    t0 = time.perf_counter()
    res = conv.convert(a.pdf, page_range=(lo, hi))
    total = time.perf_counter() - t0
    doc = res.document

    manifest, latex = [], []
    for item, _ in doc.iterate_items():
        if item.label != "formula":
            continue
        prov = item.prov[0]
        page = doc.pages[prov.page_no]
        img = page.image.pil_image
        k = img.width / page.size.width
        bb = prov.bbox.to_top_left_origin(page.size.height)
        pad = (bb.b - bb.t) * a.margin
        box = (max(0, (bb.l - pad) * k), max(0, (bb.t - pad) * k),
               min(img.width, (bb.r + pad) * k), min(img.height, (bb.b + pad) * k))
        name = f"{len(manifest) + 1:02d}.png"
        img.crop(tuple(round(v) for v in box)).save(out / name)
        manifest.append({"file": name, "page": prov.page_no,
                         "bbox": [round(v, 1) for v in (bb.l, bb.t, bb.r, bb.b)],
                         "orig": item.orig})
        latex.append(item.text)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(manifest)} crops to {out}")
    if a.codeformula:
        enrich = res.timings["doc_enrich"].total() if "doc_enrich" in res.timings else None
        (out / "codeformula.json").write_text(json.dumps(
            {"convert_sec": total, "formula_stage_sec": enrich, "latex": latex},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"CodeFormula: convert {total:.0f}s, formula stage {enrich:.0f}s for {len(latex)} formulas")


if __name__ == "__main__":
    main()
