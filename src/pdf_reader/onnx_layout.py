"""
onnx_layout.py - Docling's layout model (Heron, RT-DETR) on ONNX Runtime, without PyTorch.

Docling ships an ONNX Runtime engine for its layout model and an official ONNX export of Heron
(docling-project/docling-layout-heron-onnx), but the engine takes its image preprocessing from
transformers.AutoImageProcessor, and in transformers 5 both RT-DETR processors (torchvision and PIL) need torch.
The preprocessing of this model is small - its preprocessor_config.json says: resize to 640 x 640 with bilinear
resampling, no rescaling, no normalisation - so it is done here with PIL + numpy, and checked against the
transformers processor (bench/check_onnx_layout.py: at most 1/255 on 0.03 % of values, all 199 layout boxes of
15 pages unchanged). The label names are read from config.json directly, so transformers is not needed at all.

use_onnx_layout(opts) switches a PdfPipelineOptions to this engine; importing this module installs the
preprocessor and the label reader (the CPU-only device choice is in cpu_only.py).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from docling.datamodel.object_detection_engine_options import OnnxRuntimeObjectDetectionEngineOptions
from docling.models.inference_engines.object_detection.onnxruntime_engine import OnnxRuntimeObjectDetectionEngine

from pdf_reader.models import LAYOUT_DIR

RESAMPLE = {0: Image.NEAREST, 1: Image.LANCZOS, 2: Image.BILINEAR, 3: Image.BICUBIC}


class NumpyRTDetrProcessor:
    """The subset of RTDetrImageProcessor this model uses: resize (+ rescale/normalise if the config asks)."""

    def __init__(self, cfg):
        self.size = cfg["size"]
        self.resample = RESAMPLE[cfg.get("resample", 2)]
        self.do_resize = cfg.get("do_resize", True)
        self.do_rescale, self.rescale_factor = cfg.get("do_rescale", False), cfg.get("rescale_factor", 1 / 255)
        self.do_normalize = cfg.get("do_normalize", False)
        self.mean = np.array(cfg.get("image_mean", [0, 0, 0]), np.float32)[:, None, None]
        self.std = np.array(cfg.get("image_std", [1, 1, 1]), np.float32)[:, None, None]
        if cfg.get("do_pad"):
            raise ValueError("padding is not implemented for this processor")

    def __call__(self, images, return_tensors="np"):
        out = []
        for img in images:
            img = img.convert("RGB")
            if self.do_resize:
                img = img.resize((self.size["width"], self.size["height"]), self.resample)
            x = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # the Heron export takes uint8 pixels
            if self.do_rescale or self.do_normalize:
                x = x.astype(np.float32)
                if self.do_rescale:
                    x = x * np.float32(self.rescale_factor)
                if self.do_normalize:
                    x = (x - self.mean) / self.std
            out.append(x)
        return {"pixel_values": np.stack(out)}


def _load_preprocessor(self, model_folder: Path):
    cfg = json.loads((Path(model_folder) / "preprocessor_config.json").read_text(encoding="utf-8"))
    if cfg.get("image_processor_type") not in ("RTDetrImageProcessor", "RTDetrImageProcessorFast"):
        raise ValueError(f"unexpected image processor {cfg.get('image_processor_type')}")
    return NumpyRTDetrProcessor(cfg)


def _load_label_mapping(self, model_folder: Path):
    cfg = json.loads((Path(model_folder) / "config.json").read_text(encoding="utf-8"))
    return {int(i): name for i, name in cfg["id2label"].items()}  # what transformers.AutoConfig would give


OnnxRuntimeObjectDetectionEngine._load_preprocessor = _load_preprocessor
OnnxRuntimeObjectDetectionEngine._load_label_mapping = _load_label_mapping
_resolve_folder = OnnxRuntimeObjectDetectionEngine._resolve_model_folder
# the exact files this project ships (models.py), not whatever Docling's Hugging Face repository holds today
OnnxRuntimeObjectDetectionEngine._resolve_model_folder = lambda self, repo_id, revision: (
    LAYOUT_DIR if (LAYOUT_DIR / "model.onnx").exists() else _resolve_folder(self, repo_id, revision))


def use_onnx_layout(opts):
    """Heron layout on ONNX Runtime (needs the "layout_heron_default" preset, the only one with an ONNX export)."""
    opts.layout_options.engine_options = OnnxRuntimeObjectDetectionEngineOptions()
    return opts
