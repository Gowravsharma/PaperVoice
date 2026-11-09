"""
update_hf_readme.py - Upload a polished, professional Model Card to Hugging Face (Vibrant8849/PaperVoice).
"""
from huggingface_hub import HfApi

REPO_ID = "Vibrant8849/PaperVoice"

MODEL_CARD = """---
license: other
tags:
- onnx
- text-to-speech
- pdf
- audio
- tts
- kokoro
- docling
- ocr
pipeline_tag: text-to-speech
language:
- en
---

# PaperVoice Models (v2.0.0)

Official model repository for **[PaperVoice](https://github.com/Gowravsharma/PaperVoice)** — the open-source, 100% offline CPU research PDF reader with read-along word highlighting and mathematical formula narration.

PaperVoice automatically downloads these model weights on first start and verifies every file against the cryptographic SHA-256 checksums below.

---

## Model Inventory & Components

### 1. Document Layout & OCR Engine
* **`layout_heron/`** (Docling Heron RT-DETR): Page object detection (paragraphs, section headers, tables, display math). Converted to ONNX with pure NumPy preprocessing.
* **`ocr/`** (PP-OCRv6 small + direction classifier via RapidOCR): High-efficiency text detection and recognition for scanned research papers.

### 2. Mathematics & Table Structure Recognition
* **`ppfn_plus_s.onnx`** (PP-FormulaNet+ S): Recognizes mathematical formulas directly from page crops into LaTeX (~58M parameters, ~50× faster on CPU than CodeFormulaV2).
* **`tableformer_fast/`** (TableFormer Fast): Recovers row-and-column grid structure and cell coordinates from complex academic tables.

### 3. Neural Voice Engine (v2.0 Kokoro-82M)
* **`kokoro/kokoro-v1.0.onnx`** (StyleTTS 2 architecture, 82M parameters, 24 kHz): Pristine reference neural voice delivering natural human prosody and phase fidelity ($r = 1.000000$).
* **`kokoro/voices-v1.0.bin`**: Style embedding vectors for voice personalities (`af_heart`, etc.).
* **`kokoro/kokoro-v1.0.fast.onnx`**: Conv1D STFT-accelerated model (378 ms faster per sentence).
* **`kokoro/kokoro-v1.0.fused_snake.onnx`**: Optimized model with 48 fused Snake periodic activations (1.99 s sentence latency, RTF 0.313).
* **`kokoro/snake_op.dll`**: Zero-dependency statically linked AVX2/FMA C++ custom operator for ONNX Runtime on Windows.

*(Note: The v1 classic voice, Piper `en_US-lessac-medium`, is licensed for research-only and downloaded on demand from Piper's official repository).*

---

## Cryptographic Checksums (SHA-256)

Every file downloaded by PaperVoice is validated against these exact checksums:

| File Path | Size | SHA-256 Checksum |
|---|---|---|
| **Neural Voice (Kokoro-82M)** | | |
| `kokoro/kokoro-v1.0.onnx` | 325.5 MB | `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5` |
| `kokoro/voices-v1.0.bin` | 28.2 MB | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` |
| `kokoro/kokoro-v1.0.fast.onnx` | 325.5 MB | `bd26413cf0b1505c98b37b392fcb82ad16dbf20d81eddbbaa9f24a5a5c50b573` |
| `kokoro/kokoro-v1.0.fused_snake.onnx` | 325.5 MB | `1abf477b767d56639f154c7a66ff3b55baa32454e043e688137403ad027b94f1` |
| `kokoro/snake_op.dll` | 359 KB | `568e4c9fb3419703a06dc5c71cb4011d6eba4b1d1571c5bb9747ded1d6327660` |
| **Formula Recognition** | | |
| `ppfn_plus_s.onnx` | 318.4 MB | `8f85ff062a32f137e07dae64d6f955654333b3c0d59e74d2b728ae3907c309a9` |
| `ppfn_plus_s.tokenizer.json` | 1.5 MB | `352f9ea33f61cca2cc5d44bfa11ead41f243c4be96255f39bae9c0246787dc9d` |
| **Page Layout (Heron)** | | |
| `layout_heron/model.onnx` | 114.6 MB | `59c81a3a2923042d85034ffc487f8f47e4854117e879aef89b2b9f728fb4922a` |
| `layout_heron/config.json` | 3 KB | `c6e67b6cdf64fa245779d0409bb994aa96e8c1790e15ec07a65843628e232180` |
| `layout_heron/preprocessor_config.json` | 422 B | `54d086cf0d7d371f7fac36e5d3a0dae31e211298affc816106cc376506799d58` |
| **Table Structure (TableFormer Fast)** | | |
| `tableformer_fast/encoder.onnx` | 89.2 MB | `f8f871b0294f456c43761d503af85a0a5abf4132c705ef6f5a7ba32dd505bb26` |
| `tableformer_fast/tag_step.onnx` | 42.6 MB | `8e1856360d6ea2156cd756969915046a5d1e4744b92fcc507ffe921efe48fc37` |
| `tableformer_fast/bbox.onnx` | 42.7 MB | `0315f1cffd7acbc3a831b2de5bd44cfc1319265aafcb5d0d08f50b790d02a89a` |
| `tableformer_fast/tm_config.json` | 3 KB | `dca6762508dddfae6d57d6cb4ef822c6000119dff0f3b6489db7413118c2622a` |
| **Scanned Document OCR (RapidOCR)** | | |
| `ocr/PP-OCRv6_det_small.onnx` | 4.6 MB | `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f` |
| `ocr/PP-OCRv6_rec_small.onnx` | 16.1 MB | `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884` |
| `ocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | 1.4 MB | `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c` |
| `ocr/ppocrv6_dict.txt` | 114 KB | `b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d` |

---

## Performance on Laptop CPU (Intel Core i5, 4 Cores)

* **Time-to-First-Sound (TTFS):** 3.28 s – 4.79 s (streaming 130-char clause chunks).
* **Real-Time Factor (RTF):** 0.313 – 0.447 ($>2.2\times$ faster than real-time speech).
* **CPU Utilization:** 25% (2 dedicated intra-op threads, eliminating cache false sharing).
* **Acoustic Fidelity:** $r = 1.000000$ waveform correlation, SNR $> 95\text{ dB}$.

---

## Upstream Licenses & Attribution

* **Kokoro-82M:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0)
* **Docling Heron:** [docling-project/docling-layout-heron-onnx](https://huggingface.co/docling-project/docling-layout-heron-onnx) (Apache-2.0)
* **PP-FormulaNet+ S & PP-OCR:** [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (Apache-2.0)
* **TableFormer:** [docling-project/docling-models](https://huggingface.co/docling-project/docling-models) (CDLA-Permissive-2.0)
"""

def main():
    api = HfApi()
    user = api.whoami().get("name")
    print(f"Logged in as {user}. Updating README.md in {REPO_ID} ...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="docs: publish polished v2.0.0 model card with Kokoro neural weights",
    )
    print("SUCCESS: Model Card successfully published to Hugging Face!")

if __name__ == "__main__":
    main()
