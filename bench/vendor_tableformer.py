"""
vendor_tableformer.py - generate src/pdf_reader/vendor/tableformer/ from docling-ibm-models (torch-free parts).

Usage (main env, where docling-ibm-models is installed): uv run python bench/vendor_tableformer.py

Docling's table stage turns the TableFormer model output into table cells with TFPredictor and its helpers
(cell matching with the PDF text, OTSL decoding, post-processing). That code is plain Python/numpy, but
docling-ibm-models depends on torch, so installing it would bring torch back. The needed modules are copied
here (MIT licence, IBM - LICENSE copied alongside) with imports pointed at the copies. Only tf_predictor.py is
changed beyond imports: its PyTorch model becomes pdf_reader.onnx_tableformer.OnnxTableModel and its torch
helpers come from tf_shims.py. Every replacement must match exactly once, so a changed upstream file fails
loudly instead of producing a silently different copy.
"""
import shutil
from importlib.metadata import distribution
from pathlib import Path

import docling_ibm_models

SRC = Path(docling_ibm_models.__file__).parent / "tableformer"
OUT = Path(__file__).resolve().parents[1] / "src" / "pdf_reader" / "vendor" / "tableformer"
DIST = distribution("docling-ibm-models")
MODULES = ["settings.py", "otsl.py", "data_management/tf_cell_matcher.py",
           "data_management/matching_post_processor.py", "utils/app_profiler.py", "utils/mem_monitor.py",
           "data_management/tf_predictor.py"]
PKG = "docling_ibm_models.tableformer"
NEW_PKG = "pdf_reader.vendor.tableformer"

PREDICTOR_REPLACE = [
    ("import torch\nfrom safetensors.torch import load_model\n", "import contextlib\n"),
    (f"import {PKG}.common as c\nimport {PKG}.data_management.transforms as T\n",
     "import pdf_reader.vendor.tf_shims as c\nimport pdf_reader.vendor.tf_shims as T\n"),
    (f"import {PKG}.utils.utils as u\n", "import pdf_reader.vendor.tf_shims as u\n"),
    (f"from {PKG}.models.common.base_model import BaseModel\n"
     f"from {PKG}.models.table04_rs.tablemodel04_rs import (\n    TableModel04_rs,\n)\n",
     "from pdf_reader.onnx_tableformer import OnnxTableModel\n"),
    ("            torch.set_num_threads(self._num_threads)\n", "            pass  # threads are set per ONNX session\n"),
    ("        self._model.eval()\n", ""),
    ("            model = TableModel04_rs(self._config, self._init_data, self._device)\n",
     "            model = OnnxTableModel(self._config[\"model\"][\"save_dir\"], self._num_threads)\n"),
    ("        img = torch.FloatTensor(img / 255.0)\n        image_batch = img.unsqueeze(dim=0)\n"
     "        image_batch = image_batch.to(device=self._device)\n",
     "        img = (img / 255.0).astype(np.float32)\n        image_batch = img[None]\n"),
]
PREDICTOR_REPLACE_ALL = [
    ("with torch.no_grad():", "with contextlib.nullcontext():"),
    ("torch.argmax(outputs_class, dim=1)", "np.argmax(outputs_class, axis=1)"),
]
# the safetensors loading block (from its comment to the end of the with-block) is dropped: the ONNX graphs load above
LOAD_START = "            # Load model from safetensors\n"
LOAD_END = "        return model\n"


def header(module, changed):
    what = ("Changes: the PyTorch model is replaced by ONNX Runtime graphs (pdf_reader.onnx_tableformer) and the\n"
            "# torch helpers by numpy equivalents (pdf_reader.vendor.tf_shims); imports point at the copies.\n"
            if changed else "Changes: imports point at the copies in this folder; nothing else.\n")
    return (f"# Copied by bench/vendor_tableformer.py from docling-ibm-models {DIST.version}\n"
            f"# (docling_ibm_models/tableformer/{module}, MIT licence, Copyright IBM Corp. - see LICENSE).\n"
            f"# {what}# Do not edit by hand; re-run the script.\n")


def patch_predictor(s):
    for old, new in PREDICTOR_REPLACE:
        if s.count(old) != 1:
            raise SystemExit(f"tf_predictor: expected one match, found {s.count(old)}: {old[:70]!r}")
        s = s.replace(old, new)
    i = s.index(LOAD_START)
    j = s.index(LOAD_END, i)
    if "load_model(model, model_fn" not in s[i:j]:
        raise SystemExit("tf_predictor: unexpected model-loading block")
    s = s[:i].rstrip() + "\n\n" + s[j:]
    for old, new in PREDICTOR_REPLACE_ALL:
        if old not in s:
            raise SystemExit(f"tf_predictor: not found: {old!r}")
        s = s.replace(old, new)
    return s


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for module in MODULES:
        s = (SRC / module).read_text(encoding="utf-8")
        changed = module.endswith("tf_predictor.py")
        if changed:
            s = patch_predictor(s)
        s = s.replace(PKG, NEW_PKG)
        code = [l for l in s.splitlines() if not l.lstrip().startswith("#")]
        if any(("import torch" in l or "torch." in l) for l in code):
            raise SystemExit(f"{module}: still uses torch")
        dst = OUT / module
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(header(module, changed) + s, encoding="utf-8")
    for pkg in {OUT} | {(OUT / m).parent for m in MODULES}:
        (pkg / "__init__.py").touch()
    lic = next(f for f in DIST.files if f.name == "LICENSE")
    shutil.copy(DIST.locate_file(lic), OUT / "LICENSE")
    n = sum(len((OUT / m).read_text(encoding="utf-8").splitlines()) for m in MODULES)
    print(f"wrote {len(MODULES)} modules ({n} lines) + LICENSE to {OUT}")


if __name__ == "__main__":
    main()
