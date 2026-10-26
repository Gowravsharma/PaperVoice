"""
check_onnx_layout.py - the ONNX layout path against the PyTorch one (main env, which has both).

1. pixels: our numpy preprocessor vs transformers' RTDetrImageProcessor, on rendered pages
2. detections: layout clusters (label, box) of PyTorch Heron vs ONNX Heron, page by page

Usage: uv run python bench/check_onnx_layout.py <pdf> [<pdf> ...] --pages 1-6
"""
import argparse
import json

import numpy as np
import pypdfium2 as pdfium

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from pdf_reader.onnx_layout import NumpyRTDetrProcessor, use_onnx_layout


def converter(onnx):
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    opts.accelerator_options = AcceleratorOptions(num_threads=4, device="cpu")
    opts.layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
    if onnx:
        use_onnx_layout(opts)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
        pipeline_cls=StandardPdfPipeline, pipeline_options=opts)})


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--pages", default="1-6")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.pages.split("-"))

    # 1. pixels
    from transformers import AutoImageProcessor
    from huggingface_hub import snapshot_download
    folder = snapshot_download("docling-project/docling-layout-heron-onnx")
    hf = AutoImageProcessor.from_pretrained(folder)
    ours = NumpyRTDetrProcessor(json.load(open(f"{folder}/preprocessor_config.json")))
    worst = 0.0
    for f in a.pdfs:
        pdf = pdfium.PdfDocument(f)
        for p in range(lo - 1, min(hi, len(pdf))):
            img = pdf[p].render(scale=2).to_pil()
            x, y = hf(images=[img], return_tensors="np")["pixel_values"], ours([img])["pixel_values"]
            d = np.abs(x.astype(np.float32) - y.astype(np.float32))
            worst = max(worst, float(d.max()))
            share = float((d > 0).mean())
    print(f"preprocessing: max pixel difference vs transformers = {worst} (of 255), "
          f"share of differing values on the last page = {share:.4f}; dtype {x.dtype} vs {y.dtype}")

    # 2. detections
    conv_t, conv_o = converter(False), converter(True)
    total, same, moved = 0, 0, []
    for f in a.pdfs:
        for p in range(lo, hi + 1):
            rt = conv_t.convert(f, page_range=(p, p))
            ro = conv_o.convert(f, page_range=(p, p))
            ct = [(c.label, c.bbox.as_tuple()) for c in rt.pages[0].predictions.layout.clusters]
            co = [(c.label, c.bbox.as_tuple()) for c in ro.pages[0].predictions.layout.clusters]
            for lab, box in ct:
                total += 1
                best = max((iou(box, b2) for l2, b2 in co if l2 == lab), default=0)
                if best > 0.9:
                    same += 1
                else:
                    moved.append((f.split("/")[-1], p, str(lab), round(best, 2)))
            print(f"{f.split('/')[-1][:30]:30} p{p}: torch {len(ct)} boxes, onnx {len(co)} boxes")
    print(f"detections: {same}/{total} PyTorch boxes have an ONNX box with the same label and IoU > 0.9")
    for m in moved[:20]:
        print("   differs:", m)


if __name__ == "__main__":
    main()
