"""
memory_per_model.py - memory of ONE model loaded alone in a fresh process (run once per model).

Usage: uv run python bench/memory_per_model.py formula|formula_lean|layout_torch|layout_onnx|table|piper
layout_torch and table need PyTorch: uv sync --group export
"""
import gc
import sys

import psutil

P = psutil.Process()


def mb():
    gc.collect()
    m = P.memory_info()
    return m.rss / 2**20, getattr(m, "private", m.rss) / 2**20


def main():
    which = sys.argv[1]
    import numpy  # noqa: F401
    if which.startswith("layout_torch") or which == "table":
        import torch  # noqa: F401  (counted as the base for torch models)
    r0, p0 = mb()
    if which in ("formula", "formula_lean"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        if which == "formula_lean":
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False
        ort.InferenceSession("models/ppfn_plus_s.onnx", so, providers=["CPUExecutionProvider"])
    elif which == "piper":
        from pdf_reader.read_aloud import Narrator
        Narrator()
    elif which in ("layout_torch", "layout_onnx", "table"):
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions, PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        r0, p0 = mb()  # docling imports are counted separately
        opts = PdfPipelineOptions(do_ocr=False, do_table_structure=(which == "table"))
        opts.accelerator_options = AcceleratorOptions(num_threads=4, device="cpu")
        lo = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
        if which == "layout_onnx":
            from docling.datamodel.object_detection_engine_options import OnnxRuntimeObjectDetectionEngineOptions
            lo.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
        opts.layout_options = lo
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_cls=StandardPdfPipeline, pipeline_options=opts)})
        conv.initialize_pipeline(InputFormat.PDF)
        if which == "table":
            print("(layout + table; subtract layout_torch)")
    r1, p1 = mb()
    print(f"{which:14} +{r1 - r0:6.0f} MB rss   +{p1 - p0:6.0f} MB private   (process total rss {r1:.0f} MB)")


if __name__ == "__main__":
    main()
