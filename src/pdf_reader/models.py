"""
models.py - where the model files live, and `papervoice-models`, which downloads them (once, ~750 MB).

    uv run papervoice-models            # download what is missing, check every file's SHA-256
    uv run papervoice-models --check    # only report what is there

Every model is an ONNX file run by ONNX Runtime, and every user gets exactly the files this project was tested
with: each file is checked against its SHA-256 below, and the code loads them from this folder only (models/ next
to src/, or the folder in the PAPERVOICE_MODELS environment variable):

  layout_heron/        page layout: Docling's Heron (ONNX export by Docling, Apache-2.0)
  ppfn_plus_s.*        formula recognition: PP-FormulaNet_plus-S (Apache-2.0, PaddlePaddle), exported to ONNX here
  tableformer_fast/    table structure: TableFormer fast (CDLA-Permissive-2.0, IBM), exported to ONNX here
  ocr/                 text in scanned pages: RapidOCR's PP-OCRv6 small + direction classifier (Apache-2.0)
  voices/              the voice: Piper "lessac" medium

All but the voice come from this project's model repository on Hugging Face (MODEL_REPO). The voice is downloaded
from Piper's own repository instead: it is trained on the Blizzard 2013 Lessac data, whose licence allows research
use only and forbids passing the material on, so it is not re-hosted here. Its download is pinned to Piper's
v1.0.0 files and checked like the others. voices/<voice>.aligned.onnx is made from it on this machine (the same
model with its phoneme durations as an output, for exact word highlighting; read_aloud.aligned_voice).
"""
import argparse
import faulthandler
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("PAPERVOICE_MODELS", ROOT / "models"))
VOICES_DIR = MODELS_DIR / "voices"
FORMULA_DIR = MODELS_DIR
TABLE_DIR = MODELS_DIR / "tableformer_fast"
LAYOUT_DIR = MODELS_DIR / "layout_heron"
OCR_DIR = MODELS_DIR / "ocr"

VERSION = "1.0.0"
# This project's model repository on Hugging Face (set it once the files are uploaded; docs/RELEASING.md):
# MODEL_REPO = os.environ.get("PDF_READER_MODEL_REPO", "Vibrant8849/PaperVoice")
MODEL_REPO = "Vibrant8849/PaperVoice"
MODEL_REVISION = os.environ.get("PAPERVOICE_MODEL_REVISION", "main")
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/"

# relative path -> SHA-256 (bench/package_models.py fills it from the files it packages)
FILES = {
    "ppfn_plus_s.onnx": "8f85ff062a32f137e07dae64d6f955654333b3c0d59e74d2b728ae3907c309a9",
    "ppfn_plus_s.tokenizer.json": "352f9ea33f61cca2cc5d44bfa11ead41f243c4be96255f39bae9c0246787dc9d",
    "tableformer_fast/encoder.onnx": "f8f871b0294f456c43761d503af85a0a5abf4132c705ef6f5a7ba32dd505bb26",
    "tableformer_fast/tag_step.onnx": "8e1856360d6ea2156cd756969915046a5d1e4744b92fcc507ffe921efe48fc37",
    "tableformer_fast/bbox.onnx": "0315f1cffd7acbc3a831b2de5bd44cfc1319265aafcb5d0d08f50b790d02a89a",
    "tableformer_fast/tm_config.json": "dca6762508dddfae6d57d6cb4ef822c6000119dff0f3b6489db7413118c2622a",
    "layout_heron/model.onnx": "59c81a3a2923042d85034ffc487f8f47e4854117e879aef89b2b9f728fb4922a",
    "layout_heron/config.json": "c6e67b6cdf64fa245779d0409bb994aa96e8c1790e15ec07a65843628e232180",
    "layout_heron/preprocessor_config.json": "54d086cf0d7d371f7fac36e5d3a0dae31e211298affc816106cc376506799d58",
    "ocr/PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "ocr/PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "ocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    "ocr/ppocrv6_dict.txt": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
    "voices/en_US-lessac-medium.onnx": "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f",
    "voices/en_US-lessac-medium.onnx.json": "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0",
}
VOICE_FILES = ("voices/en_US-lessac-medium.onnx", "voices/en_US-lessac-medium.onnx.json")


def report_crashes():
    """A native library that crashes (an "access violation", exit code -1073741819) takes Python down without a
    word; this makes it print the Python line it was in, to stderr (the server's log, or the terminal)."""
    try:
        faulthandler.enable()
    except (RuntimeError, ValueError, OSError):  # no stderr to write to (pythonw without a log)
        pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status():
    """{relative path: "ok" | "missing" | "corrupt"}"""
    out = {}
    for rel, digest in FILES.items():
        p = MODELS_DIR / rel
        out[rel] = "missing" if not p.exists() else "ok" if digest is None or sha256(p) == digest else "corrupt"
    return out


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while block := r.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / 2**20:6.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def download(missing):
    hub = [rel for rel in missing if rel not in VOICE_FILES]
    if hub:
        if MODEL_REPO.startswith("OWNER/"):
            sys.exit("The model repository is not set yet: set PAPERVOICE_MODEL_REPO to the Hugging Face repository "
                     "that holds the models (see docs/RELEASING.md).")
        from huggingface_hub import hf_hub_download
        for rel in hub:
            print(f"downloading {rel} from {MODEL_REPO}")
            hf_hub_download(MODEL_REPO, rel, revision=MODEL_REVISION, local_dir=MODELS_DIR)
    for rel in missing:
        if rel in VOICE_FILES:
            print(f"downloading {rel} (Piper's repository)")
            _download(VOICE_URL + Path(rel).name, MODELS_DIR / rel)


def ensure(verbose=False):
    """Download whatever is missing or damaged, then check every file. Called by the server and papervoice-read at
    start-up, so a fresh install works without a separate step. Quick when everything is in place: it only
    checks that each file exists; the SHA-256 is checked for files it downloads (and by --check)."""
    missing = [rel for rel, digest in FILES.items() if not (MODELS_DIR / rel).exists()]
    if missing:
        print(f"first start: downloading the models into {MODELS_DIR} (once, ~750 MB)", flush=True)
        download(missing)
        bad = [rel for rel in missing if FILES[rel] and sha256(MODELS_DIR / rel) != FILES[rel]]
        if bad:
            for rel in bad:
                (MODELS_DIR / rel).unlink(missing_ok=True)
            sys.exit(f"these downloads did not verify (different from the tested files), removed: {bad}")
    elif verbose:
        print("all model files are in place")


def main():
    ap = argparse.ArgumentParser(description="Download the model files PaperVoice needs (once; the server and "
                                             "papervoice-read also do this by themselves on first start).")
    ap.add_argument("--check", action="store_true", help="only report which files are present and intact")
    a = ap.parse_args()
    report_crashes()
    print(f"models folder: {MODELS_DIR}")
    st = status()
    for rel, s in st.items():
        print(f"  {s:8} {rel}")
    if a.check:
        sys.exit(0 if all(s == "ok" for s in st.values()) else 1)
    for rel, s in st.items():
        if s == "corrupt":
            (MODELS_DIR / rel).unlink()
    ensure(verbose=True)
    from pdf_reader.read_aloud import aligned_voice
    print("voice with exact word timing:", aligned_voice("en_US-lessac-medium") or "not available (estimated timing is used)")
    print("all models are in place")


if __name__ == "__main__":
    main()
