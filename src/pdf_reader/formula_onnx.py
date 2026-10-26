"""
formula_onnx.py - PP-FormulaNet_plus-S on ONNX Runtime as Docling's formula model (CPU, no Paddle).

FormulaRecognizer      crop image -> LaTeX. Port of PaddleX's UniMERNet pre/post-processing
                       (crop margin, resize/pad to 384, normalise, tokenizer decode, normalize).
OnnxFormulaModel       Docling enrichment model that fills `formula` items with that LaTeX, tidied for speech
                       (latex_cleanup.tidy: equation numbers dropped, letter-spaced words re-joined).
FastFormulaPdfPipeline StandardPdfPipeline with OnnxFormulaModel in place of CodeFormulaV2.
make_converter()       DocumentConverter with the light CPU settings and this pipeline; every model runs on
                       ONNX Runtime, so PyTorch is not needed: layout (onnx_layout.py, Docling's official Heron
                       export), tables (onnx_tableformer.py, our export of TableFormer fast), formulas (here),
                       OCR (RapidOCR).

Model files (models.FORMULA_DIR, see models.py): ppfn_plus_s.onnx (exported with paddle2onnx, see bench/) and
ppfn_plus_s.tokenizer.json (fast_tokenizer_file from the model's inference.yml).
"""
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort

import pdf_reader.ort_lean  # noqa: F401  (before any session is created)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat, ItemAndImageEnrichmentElement
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    LayoutObjectDetectionOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseItemAndImageEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, DoclingDocument, NodeItem, TextItem
from ftfy import fix_text
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from pdf_reader import cpu_only
from pdf_reader.latex_cleanup import tidy
from pdf_reader.models import FORMULA_DIR, OCR_DIR
from pdf_reader.onnx_layout import use_onnx_layout
from pdf_reader.onnx_tableformer import install_table_stage

INPUT_SIZE = 384
MEAN, STD = 0.7931, 0.1738
EOS_ID = 2


class FormulaRecognizer:
    def __init__(self, model_dir: Path = FORMULA_DIR, threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_dir / "ppfn_plus_s.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.tokenizer = Tokenizer.from_file(str(model_dir / "ppfn_plus_s.tokenizer.json"))

    # ------------------------------------------------------------ preprocessing
    @staticmethod
    def _crop_margin(img):
        data = np.asarray(img.convert("L"), dtype=np.uint8)
        lo, hi = int(data.min()), int(data.max())
        if lo == hi:
            return img
        ink = (data - lo) / (hi - lo) * 255 < 200
        ys, xs = np.nonzero(ink)
        return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    def preprocess(self, img):
        img = self._crop_margin(img.convert("RGB"))
        w, h = img.size
        # shorter edge -> 384 (bilinear), then fit inside 384x384 and centre-pad with black
        if w <= h:
            img = img.resize((INPUT_SIZE, int(INPUT_SIZE * h / w)), resample=Image.Resampling.BILINEAR)
        else:
            img = img.resize((int(INPUT_SIZE * w / h), INPUT_SIZE), resample=Image.Resampling.BILINEAR)
        img.thumbnail((INPUT_SIZE, INPUT_SIZE))
        dw, dh = INPUT_SIZE - img.width, INPUT_SIZE - img.height
        img = ImageOps.expand(img, (dw // 2, dh // 2, dw - dw // 2, dh - dh // 2))
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        # cv2.COLOR_BGR2GRAY on an RGB array: channel 0 is weighted as blue
        gray = 0.114 * x[..., 0] + 0.587 * x[..., 1] + 0.299 * x[..., 2]
        return gray[None, None].astype(np.float32)

    # ------------------------------------------------------------ decoding
    @staticmethod
    def _normalize(s):
        """PaddleX UniMERNetDecode.normalize: drop spaces next to non-letters."""
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter, noletter = r"[a-zA-Z]", r"[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            for m in re.findall(r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})", x[0]):
                if m not in ("\\operatorname", "\\mathrm", "\\text", "\\mathbf") and m.strip():
                    s = s.replace(m, m + "XXXXXXX").replace(" ", "")
                    names.append(s)
        if names:
            s = re.sub(text_reg, lambda _: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                return s.replace("XXXXXXX", " ")

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if EOS_ID in ids:
            ids = ids[: ids.index(EOS_ID) + 1]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        text = re.sub(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}", r"\1", text).replace('"', "")
        return self._normalize(fix_text(text))

    def __call__(self, img):
        ids = self.sess.run(None, {self.input: self.preprocess(img)})[0].reshape(-1)
        return self.decode(ids)


class OnnxFormulaModel(BaseItemAndImageEnrichmentModel):
    images_scale = 2.0
    margin = 0.08  # fraction of box height added on every side
    elements_batch_size = 16
    narrow = 0.15  # a display formula box narrower than this share of the page holds only "(2.3)"

    def __init__(self, enabled: bool = True, threads: int = 4, model_dir: Path = FORMULA_DIR):
        self.enabled = enabled
        self._recognizer, self._model_dir, self._threads = None, model_dir, threads

    @property
    def recognizer(self):
        """Loaded on the first display formula, so documents without any never hold the model (~330 MB)."""
        if self._recognizer is None:
            self._recognizer = FormulaRecognizer(self._model_dir, self._threads)
        return self._recognizer

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, TextItem) and element.label == DocItemLabel.FORMULA

    def prepare_element(self, conv_res: ConversionResult, element: NodeItem):
        if not self.is_processable(conv_res.document, element):
            return None
        prov = element.prov[0]
        page = conv_res.pages[prov.page_no - conv_res.pages[0].page_no]
        bb = prov.bbox.to_top_left_origin(page.size.height)
        left, right = bb.l, bb.r
        if right - left < self.narrow * page.size.width:
            # a narrow box can be just an equation number. If another formula box shares the line, it is the
            # number of an equation already found (Fahim p. 24, "(3.25)") -> skip it. Otherwise the layout
            # model missed the equation itself (p5.pdf, "(2.3)") -> read the whole text column at that height.
            if self._on_formula_line(conv_res.document, element, page.size.height, bb):
                return None
            left, right = self._column(conv_res.document, prov.page_no, page.size, (bb.l + bb.r) / 2)
        pad = (bb.b - bb.t) * self.margin
        box = BoundingBox(l=left - pad, t=bb.t - pad, r=right + pad, b=bb.b + pad,
                          coord_origin=CoordOrigin.TOPLEFT)
        image = page.get_image(scale=self.images_scale, cropbox=box)
        return ItemAndImageEnrichmentElement(item=element, image=image) if image is not None else None

    @staticmethod
    def _on_formula_line(doc: DoclingDocument, element, page_height, bb):
        """Another formula box on the same page overlaps this one vertically."""
        page_no = element.prov[0].page_no
        for it, _ in doc.iterate_items():
            if it is element or it.label != DocItemLabel.FORMULA or not it.prov or it.prov[0].page_no != page_no:
                continue
            o = it.prov[0].bbox.to_top_left_origin(page_height)
            if o.t < bb.b and o.b > bb.t:
                return True
        return False

    @staticmethod
    def _column(doc: DoclingDocument, page_no, size, x):
        """Left/right edge of the text column containing x: text blocks on the page that span x, so a
        two-column paper does not widen into the other column (fallback: 8-92 % of the page width)."""
        edges = [(it.prov[0].bbox.l, it.prov[0].bbox.r) for it, _ in doc.iterate_items()
                 if it.prov and it.prov[0].page_no == page_no and it.label == DocItemLabel.TEXT
                 and it.prov[0].bbox.l <= x <= it.prov[0].bbox.r]
        if not edges:
            return 0.08 * size.width, 0.92 * size.width
        return min(l for l, _ in edges), max(r for _, r in edges)

    def __call__(self, doc: DoclingDocument,
                 element_batch: Iterable[ItemAndImageEnrichmentElement]) -> Iterable[NodeItem]:
        for el in element_batch:
            if self.enabled:
                el.item.text, _number = tidy(self.recognizer(el.image))
            yield el.item


class FastFormulaPdfPipeline(StandardPdfPipeline):
    """CodeFormulaV2 stays disabled (do_formula_enrichment=False); OnnxFormulaModel runs instead."""

    def __init__(self, pipeline_options):
        super().__init__(pipeline_options)
        threads = pipeline_options.accelerator_options.num_threads
        self.enrichment_pipe.insert(0, OnnxFormulaModel(enabled=True, threads=threads))
        self.keep_backend = True  # formula crops are cut from the page during enrichment


def make_converter(threads: int = 4, ocr: bool = False) -> DocumentConverter:
    """ocr=False for born-digital PDFs (fast). ocr=True (RapidOCR on ONNX Runtime) for pages without a text
    layer, e.g. scans or "Print to PDF" copies whose letters became vector outlines; without it such pages
    have layout boxes but no words. Layout: Heron, the only Docling layout model with an ONNX export."""
    install_table_stage()
    opts = PdfPipelineOptions()
    opts.layout_options = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
    use_onnx_layout(opts)
    cpu_only.install()  # last: the stage modules above are imported by now
    opts.do_ocr = ocr
    if ocr:
        ocr_files = {k: OCR_DIR / f for k, f in (
            ("det_model_path", "PP-OCRv6_det_small.onnx"), ("rec_model_path", "PP-OCRv6_rec_small.onnx"),
            ("cls_model_path", "ch_ppocr_mobile_v2.0_cls_mobile.onnx"), ("rec_keys_path", "ppocrv6_dict.txt"))}
        local = {k: str(v) for k, v in ocr_files.items()} if all(v.exists() for v in ocr_files.values()) else {}
        opts.ocr_options = RapidOcrOptions(backend="onnxruntime", **local)  # the shipped files (models.py)
    opts.table_structure_options.mode = TableFormerMode.FAST
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = False
    opts.accelerator_options = AcceleratorOptions(num_threads=threads, device=AcceleratorDevice.CPU)
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_cls=FastFormulaPdfPipeline, pipeline_options=opts)
    })
