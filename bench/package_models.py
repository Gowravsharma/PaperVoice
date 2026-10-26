"""
package_models.py - prepare the model repository for Hugging Face and record every file's SHA-256 in models.py.

Usage: uv run python bench/package_models.py [--out dist/hf-models]
Copies every model file except the voice (it comes from Piper's own repository, see models.py) into <out>, with
the same paths as in models/, plus a README.md model card (sources, licences, checksums). Upload that folder as
it is to the Hugging Face model repository named in models.MODEL_REPO (docs/RELEASING.md). Fills models.FILES
with the SHA-256 of each file, voice included, so every download is checked against the tested files.
"""
import argparse
import re
import shutil
from pathlib import Path

from pdf_reader import models

SOURCES = [  # (path prefix, what, source, licence)
    ("layout_heron/", "page layout: Heron (RT-DETR), ONNX export by Docling",
     "https://huggingface.co/docling-project/docling-layout-heron-onnx", "Apache-2.0"),
    ("ppfn_plus_s", "formula recognition: PP-FormulaNet_plus-S, exported to ONNX for PaperVoice (weights unchanged)",
     "https://github.com/PaddlePaddle/PaddleOCR", "Apache-2.0"),
    ("tableformer_fast/", "table structure: TableFormer fast, exported to ONNX for PaperVoice (weights unchanged)",
     "https://huggingface.co/docling-project/docling-models", "CDLA-Permissive-2.0"),
    ("ocr/", "OCR for scanned pages: PP-OCRv6 small detection/recognition + direction classifier (RapidOCR)",
     "https://github.com/RapidAI/RapidOCR", "Apache-2.0"),
]


def card(digests):
    rows = "\n".join(f"| `{rel}` | `{d}` |" for rel, d in digests.items() if rel not in models.VOICE_FILES)
    src = "\n".join(f"| `{p}` | {what} | {lic} | <{url}> |" for p, what, url, lic in SOURCES)
    return f"""---
license: other
tags:
- onnx
- pdf
- text-to-speech
---

# PaperVoice models {models.VERSION}

The ONNX models used by [PaperVoice](https://github.com/Gowravsharma/PaperVoice) (read research papers aloud, on a CPU).
PaperVoice downloads them by itself on first start and checks every file against the SHA-256 below.
Each model keeps its own licence:

| Files | Model | Licence | Source |
|---|---|---|---|
{src}

The voice (Piper "lessac") is not included: its training data is licensed for research only and may not be
redistributed, so PaperVoice downloads it from Piper's own repository.

## SHA-256

| File | SHA-256 |
|---|---|
{rows}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist/hf-models")
    a = ap.parse_args()
    digests = {}
    for rel in models.FILES:
        p = models.MODELS_DIR / rel
        if not p.exists():
            raise SystemExit(f"missing {p}")
        digests[rel] = models.sha256(p)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for rel in models.FILES:
        if rel not in models.VOICE_FILES:
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(models.MODELS_DIR / rel, dst)
    (out / "README.md").write_text(card(digests), encoding="utf-8")
    src = Path(models.__file__)
    s = src.read_text(encoding="utf-8")
    block = "FILES = {\n" + "".join(f'    "{rel}": "{d}",\n' for rel, d in digests.items()) + "}"
    s, n = re.subn(r"FILES = \{\n.*?\n\}", block, s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("FILES block not found in models.py")
    src.write_text(s, encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"wrote {out} ({size / 2**20:.0f} MB, {len(digests) - len(models.VOICE_FILES)} model files + README.md) "
          f"and the SHA-256 of {len(digests)} files into {src.name}")


if __name__ == "__main__":
    main()
