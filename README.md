# PaperVoice — listen to research papers, maths included

Turn a research paper (PDF) into natural speech **on your own computer**, with each word highlighted as it is read.
Equations, inline symbols and tables are spoken too, not skipped. It runs on an ordinary laptop CPU: no GPU, no
cloud, nothing leaves your machine.

> **Status: v1.0, looking for feedback.** It is used daily on maths-heavy papers. Please open an issue for
> anything that sounds wrong (see [Feedback](#feedback)).

## What it does

- **Reads the whole paper in order:** headings, paragraphs, display equations, tables, captions. It skips page
  numbers, running headers, citation brackets and, by default, the reference list.
- **Speaks maths:** equations are recognised from the page image and read with
  [Speech Rule Engine](https://speechruleengine.org/) (ClearSpeak), with pauses at natural points:
  "the norm of phi sub infinity, is defined as sup over t comma x in Q sub T of …". Inline symbols such as
  `φ`, `M(d, d)` and `a := σσᵀ` are spoken too.
- **Sounds like a person reading:** pauses between sentences, paragraphs and sections, short phrase breaks in
  long sentences (prosodic phrasing), and a slower pace for equations and maths-heavy sentences.
- **Reads tables** row by row: "Table 2. … FastSpeech: Training Time in hours fifty three point one two, …".
- **Read-along viewer:** the PDF stays as it is, the spoken sentence and word are highlighted, and a click
  anywhere starts reading from there. Speed is adjustable while it plays.
- **Browser extension** (Chrome, Edge): on a tab showing a PDF (arXiv, a journal, a file on your disk), click
  the icon to start listening.

**v1 reads symbols literally:** `σ` is "sigma" and `∧` is "wedge", as written. Reading meaning from context
(`∧` → "minimum", `a⁺` → "positive part of a") is planned for v2.

## How it works

```
PDF ─► Docling (layout: Heron, ONNX) ─► text blocks in reading order
         ├─ display equations ─► PP-FormulaNet+ S (ONNX) ─► LaTeX ─► Speech Rule Engine ─► words + pauses
         ├─ tables ─► TableFormer (ONNX) ─► rows and columns ─► spoken sentences
         └─ text ─► citations, links, symbols, numbers, dates (wetext) ─► prosodic phrasing
                                                   ─► Piper voice (ONNX) ─► audio + word timings
```

Every model runs on [ONNX Runtime](https://onnxruntime.ai/); PyTorch is not needed. A local server
(`papervoice-server`) does this work, and the viewer and extension talk to it on `127.0.0.1` only.

### How this differs from standard Docling

PaperVoice uses [Docling](https://github.com/docling-project/docling) (`docling-slim` 2.141.0, pinned exactly)
for page analysis, but not as it ships: it installs without its model extras and replaces or adjusts several
parts at runtime. Problems you see here may come from these changes rather than from Docling itself, so please
report them here first.

| Docling part | Standard Docling | In PaperVoice |
|---|---|---|
| Layout model | Heron on PyTorch (transformers) | Docling's own ONNX export of Heron; image preprocessing rewritten in NumPy (checked: same boxes on 199/199) |
| Table structure | TableFormer on PyTorch | our ONNX export of TableFormer *fast*; its post-processing copied from docling-ibm-models (MIT) without the torch parts (checked: identical on 14/14 tables) |
| Formulas | CodeFormulaV2 (~300M parameters) | PP-FormulaNet+ S on ONNX (~58M), ~50× faster on a CPU, slightly less accurate (0.927 vs 0.939) |
| OCR | RapidOCR, downloads its models itself | RapidOCR with the model files shipped here |
| PDF backend | docling_parse renders page images | the same, except pages with a huge vector drawing (> 1,500 paths), whose images come from pdfium: docling_parse's renderer crashes on them |
| Device choice, other engines | asks PyTorch which device to use; loads all table engines | CPU only; only the ONNX table engine is registered |
| Not used | picture classification and description, code enrichment, VLM pipelines | — |

All other Docling behaviour (reading order, page assembly, text cells) is unchanged. Details and the measurements
behind each change are in [docs/JOURNEY.md](docs/JOURNEY.md) (sections 23 and 26).

The full development story, with every measurement and the approaches that were tried and dropped, is in
[docs/JOURNEY.md](docs/JOURNEY.md).

## Requirements

- **Python 3.14** and [uv](https://docs.astral.sh/uv/) (uv installs Python for you)
- **Node.js 18 or newer** (for the maths speech engine and the viewer's PDF.js)
- About **1.5 GB of disk** (dependencies + models) and **0.4–0.9 GB of RAM** while reading
  (text-only pages ~0.4 GB; formulas and tables load their models on first use)
- Tested on **Windows 11** (Intel i5, 4 cores). macOS and Linux should work but are untested: reports welcome.

## Install

**Windows - no terminal needed:** on the repository page click **Code → Download ZIP**, unzip it, and
**double-click `install.bat`**. (Or, in a terminal, paste this into **PowerShell**; in Command Prompt `irm` is not
recognized:)

```powershell
irm https://raw.githubusercontent.com/Gowravsharma/PaperVoice/main/install.ps1 | iex
```

Either way it installs what is missing, with no administrator rights and nothing else on the computer touched: uv
(which brings its own Python) and, if Node.js 18+ is not installed, a private checksum-verified copy of it inside
the PaperVoice folder. It puts PaperVoice in `%USERPROFILE%\PaperVoice` (or stays in the unzipped folder), installs
exactly the tested versions of its parts, downloads the models (~750 MB, once), and lets the extension start the
reader. It ends by opening the extensions page for the last step below. Run it again to update; it skips what is
already done.

**By hand** (any system; needs [uv](https://docs.astral.sh/uv/) and Node.js 18+):

```bash
git clone https://github.com/Gowravsharma/PaperVoice.git PaperVoice
cd PaperVoice
uv sync                                  # Python dependencies
npm ci --prefix speech                   # maths speech (MathJax + Speech Rule Engine)
npm ci --prefix viewer                   # PDF.js for the viewer
uv run papervoice-setup                  # Windows: lets the extension start the reader by itself (once)
```

On its first start the reader downloads its models by itself (once, ~750 MB, all ONNX) and checks each file
against the SHA-256 of the files it was tested with, so everyone runs exactly the same models. To download them in
advance instead: `uv run papervoice-models`.

### Updating

When a new version is out, the extension's popup says so ("Version 1.0.1 is available…"). To update, **double-click
`update.bat`** in the PaperVoice folder (or run the one-line install command again), then click the reload icon on
the PaperVoice card of the extensions page. It first looks whether there is anything new and, if not, says so in a
second and changes nothing (`update.bat -Force` reinstalls anyway). Otherwise it stops the reading engine, fetches the new code (`git pull` if the
folder came from git, otherwise the repository's zip laid over it - a folder that was only cloned or unzipped and
never installed is simply installed), installs exactly the new locked versions of the
parts, checks the models, and keeps your downloaded models and prepared papers. Nothing else needs doing.

The popup's notice makes one small request to GitHub (`version.json`, a few bytes, at most every 12 hours; GitHub
sees your address like for any web page) - the only time the extension uses the internet. Nothing about you or your
papers is sent. Switch it off with the **update check** link at the bottom of the popup.

**Uninstall:** `uv run papervoice-setup --remove` in the folder, then delete the folder (and remove the extension).

**About the voice:** the default voice, Piper's "lessac", is trained on recordings licensed for **research use
only** (Blizzard 2013 Lessac data), so it is downloaded from Piper's own repository and is not for commercial use.

### The browser extension (Chrome or Edge)

1. Open `chrome://extensions` (or `edge://extensions`) and switch on **Developer mode**.
2. Click **Load unpacked** and choose the `extension` folder of this repository.
3. Optional, for PDFs opened from your disk: on the extension's **Details** page, switch on
   **Allow access to file URLs**.

The extension only asks for the current tab when you click its icon (`activeTab`), for the local server, and to
start it (`nativeMessaging`, through the small helper that `papervoice-setup` registers for your user only;
`uv run papervoice-setup --remove` undoes it).

## Use

Open a PDF in the browser (arXiv, a journal, a file on your disk) and click the **PaperVoice** icon →
**Read this PDF aloud**. The reader opens in a new tab next to the PDF.

There is no server to start: the extension starts the reading engine in the background when needed (the first
start takes ~15-30 s), and it stops by itself 20 minutes after the last reader tab is closed, to free its memory
(**Stop the reading engine** in the extension's popup stops it at once). After an update, the next click on the
icon replaces an engine that is still running the old code. Without the extension (or on macOS/Linux): `uv run papervoice-server`, then go to
<http://127.0.0.1:8765> and open or drop a PDF there.

**Ctrl+click** anywhere in the text to start reading from that sentence (a plain click selects text as usual). Keyboard: `Space` play/pause, `←`/`→` previous/next
sentence, `+`/`-` speed, `H` highlight: sentence / word / off, `Ctrl` `+`/`-`/`0` or `Ctrl`+scroll: zoom (pages are
redrawn sharp at every zoom).

### Command line

```bash
uv run papervoice-read paper.pdf --play                  # read aloud on the speakers
uv run papervoice-read paper.pdf --start 5 --speed 0.9 --save out/audio   # one WAV per page
uv run papervoice-convert paper.pdf --speech             # Markdown + JSON with LaTeX formulas, and the spoken script
                                                  # as text (out/paper.speech.txt)
```

## What it says, and why

| On the page | Spoken | Rule |
|---|---|---|
| `[12]`, `(Ren et al., 2019)` | (nothing) | citations are dropped |
| `(2.1)` | "equation two point one" | equation references |
| `https://…`, `name@site.org` | "a link", "an e-mail address" | |
| `x := y` | "x is defined as y" | |
| `M(d, d)`, `(0, T]` | "M of d comma d", "open paren zero comma T close paren" | brackets are spoken, not swallowed |
| `tᵢ = ih` (flattened by the PDF) | "t i equals i h" | letters that are not a word are spelled |
| `2/3` in a table | "two slash three" | in prose it is "two thirds" |
| `/`, `–` in a table cell | "no value" | |

## Known limitations (v1)

- **Inline maths comes from the PDF's text layer,** which flattens it: `tᵢ` and `t·i` both arrive as "ti", and
  nested brackets such as `(E|R|ᵖ)^{1/p}` can come out garbled. Display equations are recognised from the image
  and are much better.
- Piecewise definitions are read as "2 cases, Case 1: …".
- Scanned PDFs work (OCR), but are slower (~15 s per page), and the highlight is a box around the paragraph
  being read: a scan has no text in the browser to light word by word.
- English only; one voice (Piper "lessac", medium).
- The first page takes a few seconds; after that, pages are prepared ahead of the reader.

## Feedback

Please [open an issue](../../issues) with:

1. a link to the PDF (or the page, if it is not public),
2. the page and sentence,
3. what you heard, and what you expected to hear.

Mis-readings of maths, tables and symbols are the most useful reports.

## Privacy

Everything runs locally. The server listens on `127.0.0.1` only and refuses requests from other websites; PDFs
and their audio are cached in `out/cache/` on your disk (delete the folder to clear it).

## Development

- `bench/`: benchmarks, audits and the scripts that produced the ONNX models (`export_tableformer.py`,
  `vendor_tableformer.py`, `package_models.py`); `uv sync --group export` installs what they need (PyTorch).
- `bench/audit_pipeline.py`: traces one page through every stage and saves each stage's output and audio.
- `bench/test_server.py`: end-to-end check of the server, the extension's requests and the security rules.
- `docs/JOURNEY.md`: design notes, measurements, decisions and lessons.

## Licence and credits

PaperVoice is free software under the **GNU General Public License v3.0 or later** ([LICENSE](LICENSE)).

It builds on:

| Project | Used for | Licence |
|---|---|---|
| [Docling](https://github.com/docling-project/docling) | PDF analysis (modified: see "How this differs from standard Docling") | MIT |
| [Heron layout model](https://huggingface.co/docling-project/docling-layout-heron-onnx) | page layout (ONNX) | Apache-2.0 |
| [TableFormer](https://huggingface.co/docling-project/docling-models) | table structure (exported to ONNX here) | CDLA-Permissive-2.0 |
| docling-ibm-models (parts in `src/pdf_reader/vendor/tableformer/`) | table cell matching | MIT |
| [PP-FormulaNet+ S](https://github.com/PaddlePaddle/PaddleOCR) | formula recognition (exported to ONNX here) | Apache-2.0 |
| [Speech Rule Engine](https://github.com/Speech-Rule-Engine/speech-rule-engine), [MathJax](https://www.mathjax.org/) | maths to words | Apache-2.0 |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) + espeak-ng | the voice | GPL-3.0 |
| [wetext](https://github.com/pengzhendong/wetext) | numbers, dates, units | Apache-2.0 |
| [PDF.js](https://mozilla.github.io/pdf.js/) | the viewer | Apache-2.0 |
| [ONNX Runtime](https://onnxruntime.ai/), [RapidOCR](https://github.com/RapidAI/RapidOCR) | inference, OCR | MIT, Apache-2.0 |
