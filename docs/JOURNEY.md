# Journey: making math-paper PDFs speakable on a CPU

Goal: turn math-heavy research PDFs into text a TTS engine can read aloud, equations included,
running on **CPU only** (this machine: 4 physical cores / 8 threads, 15.7 GB RAM).

```
PDF ─► Docling layout ─► formula crop → LaTeX ─► LaTeX → MathML → spoken words ─► TTS
```

Log written 2026-10-22. Numbers below were measured on this machine.

---

## At a glance: where we started, where we are

### The machine (all numbers in this note)

| | |
|---|---|
| Laptop | Lenovo 20W1S1CU00 |
| CPU | Intel Core i5-1145G7 (11th gen), 4 cores / 8 threads, 2.6 GHz base |
| RAM | 15.7 GB |
| GPU | Intel Iris Xe (integrated; **not used**, everything runs on the CPU) |
| OS | Windows 11 Pro (build 28000) |
| Software | Python 3.14.7, Docling 2.141.0, docling-core 2.103.1, torch 2.14.1, onnxruntime 1.31.0, Node 24.22; benchmarks in a separate Python 3.12 venv (PaddlePaddle 3.3.1 / 3.1.1 for export) |

### Start vs now

| | At the start | Now |
|---|---|---|
| Formula model | CodeFormulaV2 (~300M params) | PP-FormulaNet_plus-S on ONNX Runtime (~58M params) |
| Time per formula | **35–70 s** (62 s mean on pages 5–14) | **~1.2–1.4 s** |
| One page, 6 formulas (`p5.pdf`) | 208–419 s (275 s with Docling defaults) | 43 s incl. loading all models |
| Pages 5–14 (10 pages, 50 formulas) | **52 min** (3,135 s) | **130 s** (Heron) / **108 s** (Egret-m), incl. model load |
| Full paper (46 pages, 202 formulas) | ~3.5 h (estimated: 202 × 62 s) | **~7 min** (411–469 s Heron, 415 s Egret-m) |
| Peak RAM, one page | 2.43 GB | **1.36 GB** |
| Peak RAM, 10 pages | — | 1.85 GB (Heron) / 1.70 GB (Egret-m) |
| Model files on disk | CodeFormulaV2 611 MB + Heron layout 164 MB | ONNX formula model 319 MB + Heron 164 MB (Egret-m: 75 MB) |
| Formula quality vs arXiv source (same 49 formulas) | 0.939 | 0.927 |
| Formulas read with spelled letters / equation numbers | 64 / 50 of 199 | 1 / 1 |
| Index superscripts read as powers ("the h-th power of v") | 125 of 199 | 38 (sampled: real powers) |
| Runtime dependencies added | — | `onnxruntime`, `ftfy` (no Paddle, no GPU) |

**Headline: the same 10 pages went from 52 minutes to under 2 minutes (~24–29× end to end, ~50× for the
formula step), using about half the memory, at nearly the same accuracy.**

Measured with `bench/measure_resources.py` (wall time includes model loading; peak RAM is the process's
resident memory). Average CPU use was 35–46 % of all 8 threads, i.e. roughly the 4 physical cores.
Run-to-run timing on this laptop varies by ~20 %.

### Where we are now

Working end to end:
1. PDF → layout boxes (Docling; Heron, or Egret-m with `--layout egret_m`)
2. formula boxes → LaTeX with PP-FormulaNet_plus-S on ONNX (`src/pdf_reader/formula_onnx.py`)
3. LaTeX tidied for speech: equation numbers out, letter-spaced words joined (`latex_cleanup.py`)
4. LaTeX → spoken words with paper-notation rules (`speech/tex2speech.mjs` + `readable.mjs`)

One command for steps 1–3: `uv run pdf-reader paper.pdf --out out/` → Markdown + JSON.

Since then: tables are read aloud (section 10), the whole paper becomes one normalised TTS script
(sections 11-12: `uv run pdf-reader paper.pdf --speech`), and pages stream so the first one is ready in
~9 s while the rest is prepared in the background (section 13: `uv run pdf-stream paper.pdf`).

Not done yet: the TTS audio itself; inline math inside text paragraphs (parked from the start); deciding the
default layout model.

Session: started 02:21, this summary 06:55 (about 4.5 hours, including long waits for the slow
CodeFormulaV2 reference runs).

---

## 1. Finding the real bottleneck

**Challenge.** Docling's built-in formula model (CodeFormulaV2, a ~300M-parameter vision-language model)
was "about a minute per page". We did not know which stage was actually slow, or whether tuning would fix it.

**What we did.** Wrote `profile_pipeline.py` to time every Docling stage for four configs on the same page
(defaults, a "light" config, light + the smaller Egret-m layout model, and light with formulas switched off).

**Finding.**

| Config | Convert time (1 page, 6 formulas) |
|---|---|
| default | 275 s |
| light | 419 s |
| light + Egret-m | 208 s |
| light, formulas off | **4 s** |

- The formula model was **95–99 %** of the time: **35–70 s per formula**.
- Tuning Docling (crop scale, margin, OCR off) did *not* make the formula model faster. Run-to-run noise
  was as large as 2×, so single runs could not rank configs.
- Side win: a tighter crop margin (`expansion_factor=0.08`) fixed a real bug: with the default margin,
  equation (2.3) came out as just `(2.3)`; with 0.08 the equation itself was read.

**Conclusion.** Tuning would not get us there; the formula model had to be replaced.

---

## 2. Getting a trustworthy answer key (ground truth)

**Challenge.** To compare models we needed the *correct* LaTeX for each formula. Only the paper's full
arXiv source (arXiv:0905.1863, Fahim–Touzi–Warin) was available, not an answer per crop.

Problems on the way:
- The test page `p5.pdf` had been saved via "Microsoft Print to PDF", which **destroyed the text layer** of
  the formulas. → We switched to the original 46-page pdfTeX PDF.
- The authors use ~200 private macros (`\R`, `\Lc`, `\Xh`, `\reff{..}` …). Models write `\mathbb{R}`,
  `\mathcal{L}`… so naive comparison would punish correct answers. Short macros like `\i` and `\n` would
  also corrupt `\int` and `\neq` with a naive find-and-replace.
- One source block often becomes several crops, or one crop covers several source lines.

**What we did.** `bench/build_truth.py`:
1. Expands the author's macros (whole command names only, so `\i` never touches `\int`).
2. Splits every display block into rows.
3. Aligns the 202 Docling crops to source rows with a dynamic-programming alignment in document order,
   comparing each crop's PDF text layer with a plain-text form of the LaTeX. It never looks at model
   output, so the answer key is not biased toward any model.

**Result.** 199 of 202 crops matched (checked by hand: weak matches were all correct). The 3 unmatched ones
were layout mistakes, not formulas (a text line, a lone "(3.25)").

**Scoring.** `bench/common.py` compares LaTeX token-by-token after normalising harmless differences
(`\mbox` vs `\text`, `\operatorname*{sup}` vs `\sup`, `t'` vs `t^{\prime}`, equation numbers, `\left`/`\right`).
The normaliser was refined twice after it turned out to penalise equivalent spellings.

---

## 3. Choosing a smaller formula model

**Candidates.** PP-FormulaNet-S and the newer PP-FormulaNet_plus-S (PaddleOCR), both ~58M parameters.

**Challenge.** The main project needs Python ≥ 3.14; Paddle lags behind new Python versions.
→ Benchmarks run in a **separate Python 3.12 venv** under `bench/`, so the main project stays clean.

**Result on p5 (6 formulas):** PP-FormulaNet_plus-S ≈ **1.2 s/formula**, nearly CodeFormula's quality.

---

## 4. Exporting to ONNX (and what did not work)

**Challenge 1: the exporter would not even start.** `paddle2onnx` crashed on Windows with
`DLL load failed … procedure could not be found` next to PaddlePaddle 3.2/3.3; PaddlePaddle 3.0 was
rejected as too old.
→ **Fix:** a second, export-only venv (`bench/.venv-export`) with **PaddlePaddle 3.1.1** + paddle2onnx 2.0.2rc3.
The export worked; the ONNX graph contains the whole decoding loop (image in → token ids out).

**Challenge 2: int8 quantization.** Dynamic int8 barely shrank the file (319 MB vs 318 MB) because most
weights sit inside the decoder loop, and it made results *slower and worse* (0.939 vs 0.986 on p5).
→ **Dropped.** fp32 ONNX was kept: identical output to Paddle and ~20 % faster.

---

## 5. The head-to-head on the full paper

Same crops, same answer key. CodeFormulaV2 was run on pages 5–14 (49 formulas, 52 minutes);
PP-FormulaNet_plus-S ONNX on all 199.

| Pages 5–14, same 49 formulas | Quality | ≥ 0.95 | < 0.80 | Time per formula |
|---|---|---|---|---|
| CodeFormulaV2 | 0.939 | 46.9 % | 2.0 % | **62.0 s** |
| PP-FormulaNet_plus-S (ONNX) | 0.927 | 49.0 % | 4.1 % | **1.15 s** |

All 199 formulas with ONNX: mean quality 0.914, **287 s in total**. Weak spot: very long formulas
(100+ tokens: quality 0.84, ~3 s each).

**≈ 54× faster for about the same quality.**

---

## 6. Plugging it into Docling without Paddle

**Challenge.** Paddle should not be a runtime dependency (and does not support Python 3.14).

**What we did.** Ported PaddleX's pre-processing (crop margins, resize/pad to 384×384, normalise, grayscale)
and decoding (tokenizer + clean-up) to plain numpy / PIL / `tokenizers`. The tokenizer was pulled out of the
model's config file into `models/ppfn_plus_s.tokenizer.json`.

- **Parity check:** the port reproduces PaddleX's output **exactly on all 199 crops**.
- Wrapped as a Docling enrichment model (`src/pdf_reader/formula_onnx.py`) that replaces CodeFormulaV2.
- **End-to-end check:** formulas from the real pipeline are identical to the benchmark crops (199/199).

**Result: the whole 46-page paper, 202 formulas, in 6.9 minutes** (formulas 272 s, layout 126 s)
with one command:

```
uv run pdf-reader test_papers/a_probabilistic_numeric_method.pdf --out out/fast
```

CodeFormulaV2 would need roughly 3.5 hours for the same paper.

---

## 7. Cleaning the LaTeX so it reads well aloud

**Challenge.** The recognizer's LaTeX was correct to the eye but bad to the ear. Through the speech engine
(MathJax → Speech Rule Engine, ClearSpeak):
- equation numbers were read first: *"2.1 negative script L to the X power …"*
- words came out letter-spaced and were spelled: `\mathrm{a n d}` → *"normal a normal n normal d"*,
  `\operatorname*{s u p}`, `\mathrm{i s~a~s u b s o l u t i o n~o f~}`
- the model sometimes dropped `\mathrm` altogether: `{l o c a l l y~u n i f o r m l y}`
- equation numbers were sometimes garbled: `\ 3.29)`, `\big(2.2\big)`, or inside an `aligned` block

**What we did.** `src/pdf_reader/latex_cleanup.py` (`tidy`), applied after recognition:
equation numbers are removed (and returned separately); letter-spaced groups are re-joined, with `~` as the
word separator; known operators become commands (`\limsup`, `\max`, `\exp`); words become `\text{ … }`.
Bare `{a n d}`-style groups are converted **only if every word is a known connecting word**, so a real
product like `{a b c}` is never touched. Single letters such as the transpose `^{\mathrm{T}}` are kept.

**Measured on the spoken output (199 formulas):**

| | Before | After |
|---|---|---|
| spelled-out letter runs | 64 | **1** |
| formulas starting with a spoken equation number | 50 | **1** |
| formulas speech could not parse | 0 | 0 |
| total words spoken | 15,403 | 14,954 |

Example: *"g normal a normal n normal d"* → *"g and"*; *"normal m normal a normal x sub …"* → *"max over …"*.

Lesson: **judge the output where it is consumed.** The LaTeX score barely moved (0.914 → 0.916) because the
scorer already ignored these spellings, but the spoken result changed a lot.

---

## 8. Teaching the speech engine research-paper notation

**Challenge.** The speech engine's ClearSpeak style reads *every* superscript as a power. In papers most letter
superscripts are labels (indices):
- `v^{h}(t,x)` → *"the h-th power of v"*; `\hat X_h^{t,x}` → *"raised to the t comma x power"*
- `\mathbb{R}^d` → *"r-d"*; `\mathbb{E}` → *"double struck E"*
- the placeholder in `F(\cdot, v)` → *"open paren times comma"*
- transpose `σ^{\mathrm T}` → *"to the normal T power"*; negative part `F_γ^-` → *"raised to the minus power"*
- `h^{-1/2}` → *"the negative first divided by second power of h"*

We compared settings: the literal MathSpeak style is accurate but mechanical (*"v Superscript h Baseline …"*);
no setting got both right.

**What we did.** `speech/readable.mjs` rewrites the math tree *before* it is read, using paper conventions:
letter/comma or parenthesised superscripts on a variable → *"superscript"*; `T` → *"transpose"*; lone `+`/`−`
→ *"superscript minus"*; fractional exponents → *"to the power …"*; blackboard letters → plain letters;
a `\cdot` between `(`/`,` and `,`/`)` → *"dot"*. Real powers (`σ^2`, `e^{…}`, `|x|^q`, primes) keep the
normal reading.

Two snags and fixes:
- Wrapping a script in a new group hid the variable from the engine, so *"v of (t, x)"* became
  *"v … times (t, x)"*. → insert an explicit function-application mark when a rewritten letter is followed by `(`.
- Letters put in a text node were read *"cap R"*. → use a plain math letter instead.

**Measured on 199 formulas:**

| | Literal | Readable |
|---|---|---|
| index read as a power | 125 | **38** (sampled: real powers) |
| *"r-d"* / *"double struck"* | 12 / 52 | **0 / 0** |
| placeholder dot read as *"times"* | 16 | **0** |
| formulas speech could not parse | 0 | 0 |
| total words spoken | 14,954 | 13,901 |

`node speech/tex2speech.mjs clearspeak --literal` still gives the engine's own reading;
`node speech/compare_reading.mjs '<latex>'` shows both side by side.

---

## 9. A faster layout model (Egret-m)

**Challenge.** With formulas fast, the layout model (Heron, 42.9M params) had become ~30 % of the run.
A smaller model could change *which* boxes get labelled as formulas, so speed alone was not enough to judge.

**What we did.** Made the layout model selectable (`pdf-reader --layout heron|egret_m`), ran the full paper
with both back to back, and wrote `bench/eval_layout.py`: it aligns each run's formulas to the arXiv source
with the same text-layer alignment as the answer key, so two different sets of boxes are judged fairly.

**Result (46 pages, same machine, back-to-back):**

| | Heron | Egret-m |
|---|---|---|
| layout time | 156 s | **84.5 s** (1.85× faster) |
| total time | 469 s | 415 s |
| formulas detected / matched to a real equation | 202 / 199 | 204 / 200 |
| source equations found | 247 of 247 | 246 of 247 |
| recognition quality | 0.916 | 0.919 |

The one miss: Egret-m merged equation (3.12) on page 19 into the paragraph above it, so it would be read
from the raw text layer (`D2gε` for D²g_ε) instead of as LaTeX.

Note: Heron's layout took 126 s in an earlier run and 156 s here; timings on this laptop vary by ~20 %
between runs, so compare models only within the same session.

---

## 10. Reading tables aloud

**Question.** Docling's table model (TableFormer, FAST) already gives a cell grid with header cells marked
(~4 s per table on this CPU). How should a table *sound*, and what has to be refined first?

**How a table is read for a listener** (who cannot look back at a column header):
caption first ("Table 3. …"), then the shape ("It has 7 rows and 4 columns: sigma, gamma, K, DTW"), then one
sentence per row starting with the row label, **every value with its column name**, and "End of table 3."
Long tables (> 12 rows) read 3 rows and say how many follow. Decided with the user: **v1 reads symbols
literally** (σ → "sigma"); expanding them from the caption ("standard deviation") is a v2 item, because a
symbol can mean different things and a wrong guess misleads.

**Approaches considered:** neural table-to-text (T5/BART) and LLM summaries can invent or swap numbers, which
is unacceptable for results tables, and are slow on CPU. WFST text normalisation (NeMo) is excellent but
needs `pynini`, which does not install cleanly on Windows. → **rules over Docling's grid, no new model.**
TableFormer ACCURATE was tried for a broken table: 3× slower (23.4 s vs 7.8 s) and still wrong, so FAST stays.

**Problems found and fixed** (`src/pdf_reader/table_speech.py`), tested on FastSpeech 2, then on
Tacotron 2 and WaveNet, which the rules had not seen:

| Problem | Fix |
|---|---|
| captions "missing" (3 of 7 in FastSpeech 2) | "(a) …", "(b) …", "Table N: …" sat as plain text after the table → picked up and marked as consumed |
| side-by-side tables merged into one grid | k sub-captions + columns divisible by k → split into parts |
| wrong row labels (TableFormer stretched one cell over 3 lines; "FastSpeech: MOS 3.83" was really FastSpeech 2) | detect the too-tall cell from cell geometry, re-read those rows from the **PDF text layer** by character centre inside (column × row band) |
| a part with broken numbers ("Method" in every row) | said honestly: "This part could not be read reliably" |
| `kurtosis (K)` → "kurtosis in thousand" | unit words only in column names, exact case, unambiguous units |
| `9.32 × 10-1` (superscript flattened) | "9.32 times 10 to the power minus 1" (only after × or ≪, so ranges are untouched) |
| spanning header repeated with every value | said once: "6 columns under Subjective preference …: LSTM, Concat, …" |
| row-group label split into "North" / "American" / "English" | joined and announced once as a group heading |
| author list detected as a table | no caption and almost no numbers → skipped |
| caption cut at "vs." | sentence splitter ignores vs., e.g., et al. |
| "30:" as a row label | numeric labels get the first column's name: "Total layers 30:" |

Snags on the way: the first row-band choice took a cell from the broken half of the table (wrong height);
a fixed margin either lost decimal points or pulled in the next row → character centres solved both.

**Result:** 14 tables in three papers read correctly or are flagged honestly; **103 ms for all 14** on top of
TableFormer. Example: *"FastSpeech 2 (Mel plus PWG): MOS 3.83 plus or minus 0.08."*

---

## 11. One script for the whole paper

**Goal.** Join everything in reading order into one plain-text script a TTS engine can read
(`src/pdf_reader/script.py`; `pdf-speech out/paper.json --pdf paper.pdf`, or `pdf-reader paper.pdf --speech`).

**Rules.** Headings as their own paragraph (ALL-CAPS headings put in title case so they are read as words);
paragraphs and list items with literal symbol words (shared `text_norm.py`, also used by tables); display
formulas through `tex2speech` (one Node call per document); tables through the table reader, whose caption
paragraphs are not read twice; figure captions read, images skipped; page headers/footers, footnotes and
the reference list skipped. The appendix after "References" is still read.

**Found while reading the first scripts, and fixed:**
- equation numbers written as plain integers (`\quad(1)`, `\eqno(2)`) were read aloud → stripped, but only
  when set apart by `\quad`/`\eqno`/`\tag`, so `f(1)` is never touched
- `(i+2.5)^{-5/2}` → *"raised to the negative 5 divided by 2 power"* → fraction rule now applies to any base
- `\infty` → *"normal infinity"* → upright marking dropped for non-letters
- *"Fast AND High-quality … Text TO Speech"* → small words lower-cased in headings
- *"a 3x speed-up"* → *"a 3 times speed-up"*

**A regression caught by the 199-formula check:** the infinity fix used a method MathJax does not have, so
31 formulas silently fell back to raw LaTeX. The script would have replaced them with *"A formula that could
not be read."* and nobody would have noticed. → fixed, and the script now reports `unreadable_formulas`.
Lesson: **a silent fallback needs a counter.**

**Result:**

| Paper | Words | Paragraphs | Formulas | Tables | Reference items skipped | Unreadable formulas | Time |
|---|---|---|---|---|---|---|---|
| FastSpeech 2 (15 pages) | 6,668 | 62 | 4 | 7 | 48 | 0 | ~5 s |
| Fahim et al. (46 pages) | 24,575 | 337 | 202 | 0 | 39 | 0 | 6.6 s |

(Times are for building the script from the converted JSON; conversion itself is in the sections above.)

Example from the math paper: *"… we consider the Cauchy problem: negative script L superscript X v minus F of
open paren dot comma v comma D v comma D squared v close paren equals 0 comma on open bracket 0 comma T
close paren times R superscript d comma"*

Still weak, as expected: **inline math inside sentences** is read straight from the PDF text layer
(*"on QT := (0, T] times Rd"*). That is the parked inline-math item.

---

## 12. Text normalisation: numbers, years, dates, citations

**Question from the user:** "is everything getting normalized perfectly?" It was not. An audit of both
scripts (`bench/audit_script.py`) found 1,390 plain integers, 239 decimals, 93 years, 96 equation references,
68 citations, 101 abbreviations, 435 symbols, 5 URLs/e-mails left as written. The speech engine turns
formula *structure* into words but leaves numbers as digits (754 in the 202 formulas).

**Decision with the user:** years, dates, phone numbers, money are the job of a real text normaliser
(semiotic classes), not of hand-written rules. Document structure is ours: citations **dropped**,
"(2.1)" → *"equation 2.1"*, URLs → *"a link"*, e-mails → *"an e-mail address"*.

**Choosing a normaliser on Windows:** NeMo text processing (WFST) needs `pynini`, which failed to build
(needs Microsoft C++ Build Tools). **`wetext`** (WeNet's pre-compiled English WFST grammars running on
`kaldifst`) installs on Python 3.14 and handles the classes asked for: *"Jan 1, 2020"* → *"the first of
january, twenty twenty"*, years → *"twenty nineteen"*, phone numbers, ordinals, thousands, decimals.

**Problems found and fixed:**
- wetext crashed on 58 of 542 paragraphs → narrowed down: square brackets (`[12]`, `[0, T]`) and a few
  math symbols. Citations are dropped before it, remaining brackets become parentheses, symbols become words;
  a sentence it still cannot parse is kept as written and counted (`normalizer_failures`).
- speed: 48 s for the math paper → only sentences with a digit or symbol go through wetext.
- `6,139 / 255.8` (two table values) was read as a fraction → spaced slash read as *"slash"*.
- dropping a citation that is part of the sentence left *"suggested in, and show"* → after a preposition it
  is read: *"suggested in reference twelve"*, *"see references three and four"*.
- inline-math leftovers in prose: `φ(t,·)` → *"phi(t, dot)"*; `̸=` (≠ split by the text layer) →
  *"not equal to"*; `<`, `>`, `∧`, `∨`, `⊕` by name; footnote markers (†, §, ✷) dropped.
- *"Vaswani et al.."* at a sentence end; an arXiv id read as *"nine hundred and five point eighteen
  sixty three"* → *"an arXiv identifier"*.

**Result (audit after):**

| Left as written | FastSpeech 2 before → after | Fahim before → after |
|---|---|---|
| numbers in digits | 486 → **0** | 1,236 → 9 |
| citations, URLs, e-mails | 98 → **0** | 68 → **0** |
| abbreviations | 86 → **0** | 15 → **0** |
| symbols | 25 → **0** | 410 → 10 |
| sentences the normaliser could not parse | **0** | 1 |

The few left in the math paper are flattened inline math (*"16 (delta{square root of 3h}…"* for
1/6(δ√3h …)), the parked item. Building the script now takes ~14 s (FastSpeech 2) and ~45 s (math paper),
most of it in wetext.

Sample: *"GT: MOS four point three oh plus or minus zero point oh seven."* (wetext keeps a written trailing
zero as "oh").

Order of the layers (asked by the user): each paragraph passes **once** through structure → symbols →
wetext. Our layers must come first: on raw text wetext fails on `[12]`, reads the citation year it should
never see, says *"(two point one)"* without "equation", and spells URLs character by character.

---

## 13. Streaming page by page (the user's game-engine idea)

**Idea (from the user):** don't prepare the whole paper before speaking. Like a game streams the chunk
around the player instead of the whole city, prepare page 1, start reading it, and prepare the next pages
in the background.

**Why it works:** preparing a page takes ~2-7 s on this CPU; listening to it takes ~3-4 minutes
(~555 words at ~150 words/min). The background worker stays far ahead of the listener.

**Why "page" and not "block":** the layout model needs the whole page image (one pass finds all blocks),
so a page is the smallest unit for that step.

**Design** (`src/pdf_reader/stream.py`, `pdf-stream paper.pdf`): a background worker loads the models once
(layout, ONNX formula model, a long-running Node speech process, wetext), then prepares pages from the
listener's position up to `prefetch` pages ahead; `get(p)` blocks only if page p is not ready, and a jump to
another page re-prioritises. Carried between pages: "inside References", and a paragraph cut mid-sentence
at the bottom of a page, which is joined to the first body paragraph of the next page.

**Problems found and fixed:**
- Node start-up cost 0.8 s per call → `tex2speech.mjs --serve` stays running, one line in, one line out.
- **A hidden bug from step 11 surfaced:** the equivalence check (serve vs batch) showed that **27 of the
  math paper's formulas had been silently dropped** since the script builder existed. MathJax could not parse
  the recognizer's malformed TeX (`\left vert`, stray `&`, unbalanced braces), the speech engine returned an
  empty string, and my checks only looked for raw LaTeX in the output, so they reported *"0 unreadable"*.
  → `tex2speech` now applies generic repairs (drop broken `\left`/`\right`, `&`, environment markers;
  balance braces; fix empty scripts) and re-parses; **26 of 27 recovered**, 1 honestly reported unreadable.
  The math paper's script grew from 25,525 to 29,728 words. Lesson: **check for empty output too, not just
  for wrong output.**
- With those formulas back, wetext took 3.5 minutes: its time grows with sentence length and spoken formulas
  run to ~370 words. Formulas only contain plain numbers, so only their number tokens go through wetext
  (cached) → script build back to ~51 s.
- The page-break join first glued a cut sentence onto the **figure caption** at the top of the next page,
  and read a **table** in the middle of a sentence. Captions and tables are floats → the sentence is finished
  first, then the floats are read. Display formulas are not skipped (the sentence really continues into
  them).
- Hyphen at the page break: *"one-to-" + "many"* keeps its hyphen, *"meth-" + "ods"* becomes "methods".

**Result** (measured in streaming mode, the worker running flat out; playback simulated at 150 words/min):

| | FastSpeech 2 (15 pages) | Fahim et al. (46 pages, 202 formulas) |
|---|---|---|
| models loaded | 7.6 s | 7.6 s |
| **first page ready** | **9.6 s** | **9.1 s** |
| slowest page | 4.7 s (2 tables) | 7.1 s (6-10 formulas) |
| all pages ready | 50 s | 227 s |
| listening time at 150 words/min | 46 min | 198 min |
| **times the listener would wait** | **0** | **0** |
| words vs whole-document script | identical (6,947) | 29,727 vs 29,728 |

Before streaming, the first word of the math paper could be spoken after ~8 minutes (conversion + script);
now after **~9 s**. The streamed text matches the whole-document script word for word except at page breaks,
where it is better (sentences joined, captions read after the sentence).

**Correction (found in step 14):** the "~9 s" is measured from when the pipeline starts, *after* Python has
imported Docling/PyTorch, which takes another ~12-13 s. From pressing Enter, page 1 is ready after ~21 s.

---

## 14. The voice: Piper on the CPU

**Choice:** Piper (`piper-tts` 1.8.0): small ONNX voices, built for CPUs. It installed as a ready-made
package on Python 3.14 / Windows (nothing to compile). Voice `en_US-lessac-medium` (63 MB, in
`models/voices/`).

**Speed on this laptop:** synthesis is **17-20× faster than real time** on its own (real-time factor
0.05-0.06) and ~7× faster (0.14-0.15) while the page worker prepares the next pages on the same 4 cores.
Speaking rate 150-180 words/minute, matching the estimate used for streaming.

**Design** (`src/pdf_reader/read_aloud.py`, `pdf-read paper.pdf --play` / `--save DIR`): three stages
at the same time, connected by queues: page worker (step 13) → Piper voice thread, **sentence by sentence**
(Piper yields one sentence at a time, so the first sentence is heard without waiting for the whole page) →
output (speakers via `winsound`, and/or one WAV per page). Pauses: 0.35 s between paragraphs, 0.8 s
between pages. The audio queue is bounded so synthesis does not run arbitrarily far ahead.

**Result** (FastSpeech 2, pages 1-2, save-only so the timing is not tied to the speakers):

| | Time |
|---|---|
| Python start-up and imports | 13.3 s |
| models (layout, formulas, voice, speech, normaliser) + page 1 + first sentence | 18.1 s |
| **first sound, from pressing Enter** | **31.4 s** |
| audio produced | 6.7 min, synthesised in 62 s |
| gaps heard by the listener | **0** |

The first run on the speakers (played for the user) took 53 s to the first word: a cold start right after
installing (files not yet in the disk cache).

---

## 15. "Why is it skipping the paragraphs?" – testing on every PDF

**What the user heard:** for `p5.pdf`, only the formulas were spoken; every paragraph, including a long one,
was skipped.

**Cause:** `p5.pdf` was saved with "Microsoft Print to PDF", which turned every letter into vector outlines:
**0 characters of text, 1,519 drawn shapes**. The pipeline runs with OCR off (fast, for born-digital papers),
so Docling found the paragraph boxes but no words in them, and the script builder skipped the empty
paragraphs **silently**. The formulas were still read because the formula model reads the page *image*.
This had been visible since step 2 (the formula boxes of p5 had no text) but was never followed up, and
p5 was never run through the speech pipeline: **a testing gap.**

**Fixes:**
- **OCR only where needed:** every page's text layer is checked with pdfium (milliseconds); a page with
  fewer than 20 characters is converted with RapidOCR (ONNX Runtime, CPU), loaded the first time it is
  needed. Normal pages stay fast; the OCR page took ~15-17 s instead of ~3 s.
- **No more silent skipping:** text blocks that come out empty are counted (`empty_text_blocks`) and shown
  as a warning per page.
- **Equation (2.3) was missing too:** the layout model boxed only the number "(2.3)" (noted in the original
  handoff); the recognizer returned nothing and the empty string was spoken as *"question mark"*. Now a
  formula box narrower than 15 % of the page is widened to its text column (the column it sits in, so
  two-column papers are safe) - **unless** another formula box shares the line: then it is just the number
  of an equation already found (Fahim p. 24, "(3.25)"; widening would have read that equation twice) and
  is skipped quietly. Empty formulas are reported as unreadable, never as "question mark".

**Then every PDF in the project was checked** (text layer per page; empty blocks; unreadable formulas;
audit of what is not plain words). Only `p5.pdf` lacks a text layer. Two papers never run before were
streamed (measures of maximal entropy, 20 pages; Hyperbolic cross-approximation, pages 1-20): 179 formulas,
**0 unreadable, 0 empty text blocks**, slowest page 9.6 s. They exposed:
- symbols never seen before (⊂ ∅ ◦ ⇢ ↦ ≍ ≲ ⊥ ↪, a combining hat ×40, the "theta symbol" ϑ) - ϑ and the hat
  made wetext fail on 11 sentences. → a general rule instead of a growing list: any leftover symbol is read
  by its **Unicode name** (literal, v1), with standard spoken names where that name is odd ("maps to",
  "circle", "perpendicular to"); combining accents dropped; end-of-proof squares dropped.
- a set-minus sign stored as a plain backslash → "set minus"; quotation marks, which wetext escapes as
  `\"`, are removed (they are not spoken).

**Result:** p5 now reads all of its text (590 words, all 6 formulas including (2.3)); Hyperbolic pages 1-20:
nothing left unnormalised; measures paper: 3 symbols and 4 unparsed sentences left out of 13,725 words.

Lesson: **test every input you have, and make every "skip" visible.** Both big misses in this project
(27 dropped formulas, p5's paragraphs) were silent skips.

---

## 16. Page numbers: start anywhere, always know where you are

**Asked by the user:** a page-number feature, to make testing easier and to choose where reading starts.

- `--start 12` reads from page 12 to the end; `--pages` takes "5", "3-7", "2,5,9-11" or "7-" (in the given
  order, duplicates and out-of-range pages dropped, an empty selection is an error). Works for `pdf-read`
  and `pdf-stream`.
- The background worker prepares **only the selected pages, in that order**, so `--pages 24,5` spends no
  time on pages 1-23: page 24 was heard 16.7 s after the pipeline started, then page 5, no gaps.
- Each page starts with a spoken *"Page twenty four."* and a console line `▶ page 24 of 46`, so when
  something sounds wrong it is clear which page to look at. `--no-page-numbers` turns the spoken part off.
- A jump between non-consecutive pages resets the carried state (unfinished paragraph, "inside
  References"), because the pages in between were not read.

---

## 17. Speed knob, and the local server behind a browser viewer (step 1 of 3)

**Speed:** the user found the voice fast - measured, the default is ~183 words/min (not the ~150 assumed in
step 13). `--speed` sets Piper's phoneme length (`length_scale = 1/speed`), so pitch stays natural:
0.75 → 145, 0.85 → 165, 1.0 → 183, 1.25 → 202 words/min.

**Goal (from the user):** read along in Edge with the spoken sentence highlighted, click anywhere to start
reading from there, seamless, "indexing so it is near constant time".

**Two constraints shaped the design:** Edge's built-in PDF viewer cannot be scripted by extensions (no text
positions, no highlight, no click position) → the viewer will be our own, on PDF.js. The reading engine is
Python with 300+ MB of models → it runs as a **local server** (`pdf-reader-server`, 127.0.0.1 only) and the
page in the browser talks to it.

**The index** (`src/pdf_reader/index.py`): every page becomes blocks in reading order (heading, text, caption,
formula, table), each with its position (PDF points) and its sentences; each sentence keeps its **printed
text** (to find and highlight) and its **spoken text** (for the voice). Same rules as the script builder; on
FastSpeech 2 the indexed reading matched the whole-document script except for improvements (one more citation
list dropped, "one-to-many"). A click becomes (page, x, y) → block at that point (or the next one below it in
the same column) → the sentence containing the clicked line's text.

**Sentences cut by a page break, done safely:** before, the unfinished sentence was moved into the next page;
with seeking and out-of-order indexing that next page may never be prepared next and the fragment would be
lost (another silent skip). Now page p keeps it as its `tail`; it is joined into page p+1 only if p+1 is
prepared right after p, and the reader says the tail only if it was not joined - never lost, never said twice.

**Server** (`src/pdf_reader/server.py`, standard library only): open a PDF (bytes or path; id = hash of its
content), page blocks, locate, reading sequence across pages, and WAV per sentence with the next sentences
synthesised in the background. Models load once and are shared by all open PDFs; every indexed page is saved
to `out/cache/<id>/`, so reopening a paper is instant; the whole paper is indexed in the background outward
from where the reader is.

**A bug found by the jump test:** jumping to page 40 took **173 s**. Every internal page read (the audio
look-ahead, the page-break check) also moved the indexer's priority back to the current page, so page 40 was
reached only page by page. → only what the user does (open a page, click) sets the priority; internal reads
just wait.

**Result (math paper, then the 185-page Hyperbolic paper):**

| | Time |
|---|---|
| open a PDF | 0.03-0.06 s |
| reopen after a restart (all pages from the disk cache) | 0.03 s, page in 6 ms |
| **click → sentence (locate)** | **2-75 ms** |
| reading sequence (next sentences, across pages) | 11 ms |
| audio of the first sentence after a click | 0.7-1.5 s |
| **audio of the following sentences (prefetched)** | **5-6 ms** |
| jump to a page **not indexed yet** | 4-12 s until sound (the page being prepared + that page) |
| jump back to an indexed page | **~0 s** |

Decided with the user: slimming Docling down (it brings ~1.4 GB of packages, mostly PyTorch, for the layout
model and TableFormer) comes **later**. Done now: the unused CodeFormulaV2 model (611 MB) was removed. Two other
cached models (an embedding model and an image classifier) turned out to belong to other projects
(downloaded before this one, not referenced by Docling) and were left alone.

---

## 18. The read-along viewer in Edge (step 2 of 3)

`uv run pdf-reader-server`, then open **http://127.0.0.1:8765** in Edge (or `…/?path=C:/papers/x.pdf`).

- **PDF.js** (installed locally in `viewer/`, not from a CDN - extensions may not load remote code) draws the
  pages and a text layer with the position of every printed line. Pages are drawn only when near the view,
  so a 185-page paper opens instantly.
- **Click anywhere** in the text: the click position (converted to PDF points) and the clicked line's text go
  to `/locate`; reading jumps there. **The spoken sentence is highlighted line by line**: the viewer finds the
  sentence's letters among the text-layer lines inside its block (start and end separately, so sentences
  joined across pages still light up). Formulas and tables, whose spoken words do not map back to printed
  characters, are highlighted as a whole box. The view follows the reading (switchable).
- **Controls:** play/pause (Space), previous/next sentence (← →), speed slider 0.6-1.6 (− +; applied from the next
  sentence, synthesised at that speed so the pitch stays natural), Follow, Open / drag-and-drop a PDF.
- The current and next two sentences' audio are fetched ahead; the reading order comes from the server.

**Tested by driving headless Edge through its DevTools protocol:** the 46-page paper opened in 0.5 s; a click on
a line of page 5 highlighted exactly the sentence containing it, across its three lines, **0.3 s after the
click**; audio for that sentence and the next two was requested; no console errors.

**A false alarm, checked properly:** in headless Edge the audio stopped after 2-5 s. Recording every `pause()`
call showed that the viewer never paused it; headless Edge suspends audio in an invisible tab. With
background throttling off, the first sentence (14.1 s) played to the end and the reader moved on to the next
sentence by itself.

---

## 19. Bug: "clicking page 2 reads page 5", and word-by-word highlighting

**The bug (found by the user on the Fahim paper):** a click on page 2 started reading at the first word of
page 5. The cached index showed pages 2-4 with **0 blocks**, all their items "skipped as references". The
server indexes out of order (from where the reader is to the end, then from the start); after the last pages
(the References section) the "inside References" flag was still on when it wrapped round to page 2.
→ the flag is now taken from **the page just before** (stored per page). A page indexed on a guess (its
predecessor not ready, e.g. after a jump to page 46) remembers the guess; when the page before is finished
and proves it wrong, the page is indexed again - page 46, which *is* the reference list, was being read
until this second fix. Old caches are rebuilt automatically (index version 3).

**Word-by-word highlighting (asked by the user: not the whole paragraph; the word, and in an equation the
symbol being read):**
- *When is each word said?* The Piper voice returns no timings (checked: no alignment output), but it
  returns the phonemes per sentence with spaces between words. `timing.py` shares each sentence's audio
  length over its phonemes (long vowels and the pauses at commas/full stops weigh more): an estimate.
- *Which printed word is it?* The spoken text differs from the print ("[12]" → "reference twelve", "σ" →
  "sigma"), so each printed word is turned into what it sounds like and the two word lists are aligned.
- *Where is that word on the page?* The viewer finds the sentence's letters among the text-layer characters
  of its block and marks the exact characters of the current word (a line-wrapped word gets one box per line).
- *Equations:* spoken words are matched to the printed symbols in reading order with a best-match alignment
  ("phi" → ϕ, "equals" → =, "absolute" → |, "sup" → the s of "sup", a variable → that letter); structural words
  ("sub", "the fraction with numerator") keep the previous symbol lit.
- Tables and pages without a text layer (p5.pdf) fall back to a light box.

**Checked in Edge (DevTools-driven):** click on page 2 → reads page 2. Paragraph: every highlighted word was the
spoken word (*some, conditions, stochastic, representation, …*). Formula on page 5: *absolute → |, phi → ϕ,
infinity → ∞, colon → :, equals → =, sup → s, open paren → (, t → t, comma → ,, x → x, close paren → ),
member → ∈, Q → Q*; 117 of 121 spoken words placed on one of the formula's 80 printed symbols. No errors.
The timing itself is an estimate and can only be judged by ear.

---

## 20. "The voice stops for 10-15 s when the next page comes"

**Measured instead of guessed:** the viewer now keeps a timeline per sentence (asked for, ready, started, the
silence before it). A clean reading across a page break had 0.09 s of silence, so the pause needed the
conditions of the user's session: the index had just been rebuilt (version bump) and automated tests were
competing for the CPU, so indexing ran at ~45 s per page.

**Two weaknesses turned "indexing is behind" into silence:**
- To fetch the audio of a sentence on page p+1, the server computed the reading order from it, which decides
  whether p+1 hands its last half-sentence to p+2 - so it **waited for page p+2 to be indexed** although the
  sentence was already known. With p+1 and p+2 both still to index: two page times, ~10-15 s. → audio now
  looks the sentence up on its own page (`Session.item`).
- The indexer prioritised the page last **clicked**, not the page being **listened to**. → every audio request
  now tells it where the listener is (`focus`), so the current page and the ones after it come first.
- Also: the viewer prefetches 3 sentences ahead (was 2) and asks for the next stretch of the reading order
  12 sentences before the end (was 6).

**Worst case, re-measured** (fresh index, reading started two sentences before the end of page 10, where pages 10
and 11 both end mid-sentence): **3.5 s** at the page break (the time to index page 11, which reading reached
only seconds after opening); every other gap 0.04-0.12 s. In normal reading the next page is indexed long before
it is needed (a page takes ~5 s to index and ~3 minutes to listen to).

---

## 21. "Opening a PDF takes time now; earlier it was instant"

Measured with a running server: opening takes 0.25-0.7 s, the viewer files ~0.2 s. The wait was **before the
server was ready**: Claude Code had stopped the server because the laptop ran critically low on memory; after a
restart it reloads ~3 GB of models (13 s normally, longer while Windows is paging: 79-86 % RAM used, 1.9 GB
in the page file), and the viewer **waited for that before even showing the PDF** - a design flaw.

**Fixes:**
- The viewer shows the PDF at once (PDF.js needs no server); the connection to the reading engine is made in
  the background, and a click made meanwhile starts reading as soon as the engine is ready. Measured: pages
  visible after 0.9 s, the engine had the document after 0.4 s, a click started reading.
- A failed audio request was cached as failed, so playback would stop at that sentence for good; it is now
  dropped from the cache and retried (up to 4 times, with a message while waiting).
- The server indexed **every** open paper in the background (four were open), wasting CPU and memory; now only
  the paper being read is indexed, the others pause until used again (needs a server restart).

Memory is the real constraint: the server holds ~3 GB (mostly PyTorch for Docling's layout and table models)
on a 15.7 GB laptop shared with Edge and other apps. This makes the planned "slim down" (ONNX instead of
PyTorch) more important.

---

## 22. Natural narration: audit first, then math phrasing, a narration planner, voice tuning

The user found the speech rushed and the math mechanical, and brought a plan from another LLM (which knew the
pipeline but not the code). Followed it, adapting where the code differed: Docling finds only display math
(inline math is flattened text - the parked item); Piper's "sentence silence" is a CLI option, not part of the
Python API we use; and the plan's "Mode B" (∧ → "minimum", a⁺ → "positive part") is the meaning-from-context
reading the user deferred to v2, so the faithful reading stays the default.

**Audit first** (`bench/audit_pipeline.py`, artifacts 01-06 + audio for page 5 of the math paper): 173 words/min;
**0.18 s between sentences** (pdf-read sent a whole paragraph to Piper, whose sentence chunks were joined back to
back); 0.35 s between paragraphs; Piper does pause at commas (measured: comma 0.16 s, colon 0.19 s, em dash 0.25 s,
semicolon 0.30 s, full stop 0.40 s); math reached Piper as **one breathless chunk of up to 121 words** with no
punctuation ("... colon equals sup over open paren t comma x close paren is a member of ...").

**Speech-engine settings compared** (`speech/sre_compare.mjs`, 17 expression types): ClearSpeak stays.
MathSpeak is mechanical ("StartFrac ... EndFrac"); `Fraction_Over` is ambiguous ("1 over 2 a"); `Paren_Silent`
sounds best but **drops absolute values, norms, inner products and interval ends** - a meaning change, rejected.

**Math adapter** (`speech/phrasing.mjs`), each rule checked on all 199 formulas of the math paper:
- wording for symbols with one meaning: ":=" → "is defined as", "∂" → "partial", ‖·‖ → "the norm of", ↦ → "maps
  to", the arrow under a limit → "approaches"; the formula's final comma/full stop is not read out;
- silent brackets for plain arguments ("phi of t comma x"), only if nothing is multiplied right after them and
  only if **every** use of that function in the formula can lose its brackets;
- breathing points: the formula is split at its top level (relations, separating commas, long + / − terms;
  never inside brackets or |bars|) and pauses are put there: a comma (0.16 s) or a semicolon (0.30 s).
- **Three approaches were rejected on the way, each caught by the 199-formula check:** pause tokens inserted
  into the MathML changed the engine's reading ("a plus b" → "**positive** b"); the engine's own SSML breaks
  have no depth and fell inside small pieces ("X hat sub h, superscript t comma x"); reading the pieces
  separately changed words, because the engine decides "delta **of** t" vs "delta **times** t" from the whole
  formula (4 formulas), and garbled bars lost "the absolute value of" (1). → **the words now always come from
  reading the whole formula once; the pieces only give the positions of the pauses**, matched onto that
  reading. Remaining word changes were checked one by one: brackets dropped, the reworded symbols, and two
  corrections ("w superscript epsilon times (t, x)" → "of t comma x").
- Result on 199 formulas: commas 69 → 348, semicolons 0 → 97, "open paren" 664 → 598, same single unparsable
  formula, no multi-line formula lost its "Line 1/2" structure.

**Narration planner** (`src/pdf_reader/narration.py`): one Piper call per unit (sentence, formula, heading,
table sentence) and an explicit pause class after it - clause 0.15 s, sentence 0.35 s, paragraph 0.70 s,
section 1.00 s, added to Piper's own ~0.18 s and divided by the speed. A lead-in ending in ":" ("we define:")
or "Given a map" gets a clause pause before its formula; a formula followed by a lower-case continuation
("of the diffusion X") gets a clause pause after it. Sentences are never split or reworded here.
Measured on page 5: between sentences 0.18 → **0.55 s**, clause **0.38 s**, paragraph 0.35 → **0.82 s**;
synthesis of the page 33 s → 10 s (Piper is faster on short units).

**Voice tuning:** the voice's defaults are noise_scale 0.667, noise_w 0.8, length_scale 1.0. Math is spoken
faster than prose (199 words/min at speed 1.0, because its words are short). Blind samples of one excerpt
(`bench/make_samples.py`, `out/samples/sample_A-F.wav`, key in `KEY.json`): speed 1.0 / 0.85 / 0.75,
rhythm variation noise_w 0.6 / 1.0, and formulas read slower than text.

**User's listening feedback** (speed itself is the user's own setting, so the samples were beside the point):
symbols like "phi", "mu", "sigma" were rushed; math felt too fast to follow; "With **a** :=" lost the "a"; "M(d,d)"
lost its comma and other characters went missing; long sentences sounded like chanting. Fixes, all general rules:
- **Missing characters, measured:** the voice silently skips brackets and commas inside flattened inline math,
  and says a lone "a" as the article "uh". `text_norm.inline_math`: "f(x, y)" → "f of x comma y", "(0, T)" /
  "{a, b}" → "open paren 0 comma T close paren" (only lists of short items, so ordinary parentheses in prose
  are untouched); the variable a next to an operator or inside a list → "ay" (also in formula readings,
  except the engine's own "is a member of"). "Ω" written as the OHM SIGN lookalike was not spoken → NFKC first.
- **Slowing one word is impossible with this voice:** lengthening or doubling the vowels of "phi" in the
  phonemes changed its duration by under 0.1 s (fixed-noise test); only the global pace works. So: a short
  pause before a symbol that follows the noun it names ("a smooth function, phi"), and the whole sentence read
  slower in proportion to its share of math words (up to 25 %); display formulas at 0.8 of the user's speed,
  with a 0.30 s "formula" pause class around them.
- **Prosodic phrasing** (`src/pdf_reader/prosody.py`), from the literature on where speakers pause: Gee &
  Grosjean 1983, Bachenko & Fitzpatrick 1990, and the "chinks 'n chunks" rule of Liberman & Church 1992 used in
  Festival (Taylor & Black 1998). In a long sentence a break (a comma) goes before a function word that follows
  a content word, when the phrase so far is long enough and enough is left: soonest before clause words
  (which, where, because, while, ...), later before prepositions, latest before and/or; never before "of",
  inside math, or before a bracket. Acronyms (TTS, LN) are not symbols. Checked on samples from two papers.
Page 5 now: 157 words/min while speaking (was 176), math-dense sentences slower than plain ones.

Second listen: "ay" is pronounced "eye" by this voice (checked in its phonemes) → the variable a is written
"eigh" (/eɪ/). Flattened subscripts and products: "ti = ih" (t_i = i·h) was heard "tie equals eye", losing the h
→ two lower-case letters next to an operator that are not an English two-letter word are spelled out ("t i
equals i h"). ". . ." between commas in a list ("i = 0, . . . , n") was swallowed → "dot dot dot". Lesson:
**check every respelling in the voice's phonemes**, not by how it looks.

Tables (user's question "how did it know Table 1 has two parts?"): a general rule, not per-PDF - k sub-captions
"(a) … (b) …" and a column count divisible by k. "/", "-", "n/a" as a whole cell → "no value". Fixed: "4.5/5"
was "four.five fifths" and "FastSpeech 2/2s" "two halves seconds"; in table cells a slash between numbers is
now always "slash" ("2/3" → "two slash three"), in prose a plain fraction stays a fraction. "FastSpeech 2s" →
"two S" (a version letter after a CamelCase name; the normaliser reads "2s" and "2 s" as seconds, not "2 S").

---

## 23. v1.0: no PyTorch, a browser extension, ready for GitHub

The user declared the speech good enough for **v1.0**, to publish on GitHub and LinkedIn for feedback and to use
on their own paper for a week, and chose: a browser extension; the **full slim-down (no PyTorch)** even if it
delays the release; GPL-3.0 (the voice, Piper + espeak-ng, is GPL). The code was copied to
`pdf_reader_backup_v1.0-pre-onnx_2026-10-23/` first. Git operations are left to the user.

**Where the memory went** (`bench/memory_breakdown.py`, page with 2 tables): importing torch 175 MB, Docling's
imports another 174 MB, loading the models 791 MB; 1,308 MB resident after two pages.

**Layout on ONNX.** Docling already ships an ONNX Runtime engine for its layout model and an official ONNX export
of Heron (`docling-project/docling-layout-heron-onnx`, Apache-2.0). Two obstacles: its image preprocessing comes
from `transformers.AutoImageProcessor`, whose RT-DETR processors (torchvision and PIL alike) need torch in
transformers 5; and Docling asks torch whether CUDA exists before picking a provider. → `onnx_layout.py`: the
preprocessing written out from `preprocessor_config.json` (resize to 640 × 640, bilinear, uint8 - the export takes
uint8, no rescaling), the label names read from `config.json`; `cpu_only.py` answers Docling's device question
with "cpu" (the OCR stage asked too). Check (`bench/check_onnx_layout.py`): pixels differ from transformers' by at
most 1/255 on 0.03 % of values; **199/199 layout boxes** of 15 pages from 3 papers have an ONNX box with the same
label and IoU > 0.9.

**Tables on ONNX: our own export** (no ONNX TableFormer exists). TableModel04_rs (docling-ibm-models) became three
graphs (`bench/export_tableformer.py`): image encoder (ResNet-18 + transformer encoder), **one tag-decoder step**
with its cache as input/output (the original already recomputes only the last token), and the cell-box head with
all cells in one batch (each cell is independent in the original loop). The greedy decoding loop, its two tag
corrections and the merging of spanned cells were rewritten in numpy (`onnx_tableformer.py`).
- **Two export bugs, both caught by comparing against PyTorch:** the tracing exporter baked the key length of
  `nn.MultiheadAttention` into a reshape (the step only worked at the export length) → the torch.export-based
  exporter with symbolic lengths; and the wrapper modules were left in training mode, so **dropout** went into
  the graph (4 of 14 tables differed, boxes off by 0.15) → `.eval()`. Lesson: **compare each graph alone** -
  after the fix every graph matches PyTorch to ~1e-5.
- Result on **14 real tables** (FastSpeech 2, Tacotron 2, WaveNet): **identical tag sequences 14/14**, 375 cell
  boxes within 7e-7, no class changed.
- The cell matching and post-processing (TFPredictor and helpers, ~4,200 lines, MIT) are plain Python but their
  package depends on torch → copied into `src/pdf_reader/vendor/tableformer/` by `bench/vendor_tableformer.py`,
  changing only the model lines (exact-match replacements that fail loudly if upstream changes), helpers in
  `vendor/tf_shims.py`. Docling's table-engine plugin list also imports two torch engines (TableFormer v2,
  Granite Vision) → replaced by our one engine (`install_table_stage`).

**End-to-end equivalence** (the decisive test): the spoken text of every unit, PyTorch build vs ONNX build -
FastSpeech 2 pages 6-8 (5 tables) and the math paper pages 4-5 (formulas): **152/154 identical**; the two others
differ only by the "2/2s" fix made after the reference was recorded. OCR (scanned p5.pdf) works too.

**Then memory, measured after each change** (resident, after converting the page twice):

| Change | Table page | Math page | Text page |
|---|---|---|---|
| PyTorch build | 1,308 MB | | |
| ONNX everywhere, torch not imported | 1,049 MB | | |
| ONNX Runtime arenas off (`ort_lean.py`: buffers were kept between runs, +90 MB per page) | 877 MB | | |
| Table model loaded on the first table | 878 MB | 727 MB | |
| Formula model loaded on the first formula | 567 MB | 732 MB | 403 MB |
| transformers uninstalled (no longer needed) | 547 MB | 718 MB | |

Not done: int8 weights. Earlier (section 4) dynamic int8 barely shrank the formula model (most weights sit inside
the decoder loop) and was slower and worse; not repeated. **Whole server** (`bench/test_server.py`, with the Node
speech helper, reading a math paper): **~3 GB → 630-690 MB**; ready in 8-15 s. Install: `.venv` 1.4 GB → 694 MB.

**Packaging.** `pyproject.toml`: `docling-slim[format-pdf,convert-core,feat-ocr-rapidocr]` pinned exactly (we
replace parts of its internals), no torch; the export tools in an optional group (`uv sync --group export`).
`pdf-reader-models` downloads and checks (SHA-256) the models: our two exports as one release archive
(`bench/package_models.py`, 322 MB; licences Apache-2.0 and CDLA-Permissive-2.0 allow it), the voice from
Piper's own repository (checked byte-identical to ours), Heron from Hugging Face. Model paths in one place
(`models.py`, overridable with `PDF_READER_MODELS`). README, GPL-3.0 LICENSE, `docs/RELEASING.md`.

**Browser extension** (`extension/`, Manifest V3, Chrome/Edge): click the icon on a tab showing a PDF → the
popup fetches it (the `activeTab` permission, only on click; cookies included for journals), sends it to the local
server and opens the viewer at `?doc=<id>&name=<file>`. When the server is not running, the popup says how to
start it; for PDFs on disk, it explains the "Allow access to file URLs" switch.

**Security (found while building the extension).** The server sent `Access-Control-Allow-Origin: *` and
`/api/open` accepted a local file path: any website could have opened a PDF from the disk through it and read
the text. Now: only requests addressed to `127.0.0.1`/`localhost` on its port (DNS rebinding refused), only from
the viewer itself, an extension or a local program (no Origin header); no CORS headers at all; non-PDF bodies
refused. `bench/test_server.py` checks all of this plus the extension's and viewer's requests: **12/12 pass**.

---

## 24. Highlight out of step with the voice (user report, MTL-LoRA paper)

"The yellow highlighter lags behind, sometimes it is 2-3 words ahead; once it jumped a whole sentence ahead while
the voice was still behind, and came back when that sentence ended." Three causes, all found and fixed:

1. **When a word is spoken was an estimate.** Piper gave no timings, so `timing.py` shared each sentence's
   duration over its phonemes by count. But the voice model computes an exact duration per phoneme (VITS
   duration predictor); Piper's `patch_voice_with_alignment` exposes it as a second output. The voice is now
   patched once (`<voice>.aligned.onnx`, made by `read_aloud.aligned_voice` / `pdf-reader-models`) and
   `word_times_aligned` adds up each word's samples - the durations sum to the audio length exactly. On one
   sentence the old estimate was off by up to **0.34 s** (more than a word). Where the voice joins or splits
   words ("for a" is one word to it), each written word takes the spoken word at the same relative position.
2. **Where a word is on the page was assumed.** The viewer found the sentence's letters in the PDF.js text layer
   and assumed its words follow without gaps, so a symbol only one text had (‖, superscripts, footnote marks,
   ligatures) shifted every later word; if the whole sentence was not found, it anchored on the last 24 letters.
   A sentence begun on the previous page has only its end on this page → the start landed at the top of the
   block, a whole sentence ahead - the reported jump. Now each word is looked for near where the previous one
   ended, starting after the previous sentence of the block; words from the previous page stay unlit; NFKC on
   both sides ("ﬁ" = "fi"). `bench/check_highlight.py` replays both methods on every sentence: on 4 papers
   (1,344 sentences) the old one put **22 sentences on the wrong spot and missed 22**, the new one none.
3. **The next sentence's timing started before its audio.** It is now set together with the new audio.

**Second report (same paper): at the last sentence of page 1 the highlight jumped to the end of page 2's first
paragraph.** That sentence continues on page 2, so its first words ("A prominent PEFT approach is …") are not in
that paragraph; the fix above then started at the first word it could find, "approach" - which occurs again at
the paragraph's end ("these approaches …"). My own check had not caught it: it scored the new method against
its own positions. → the check now scores honestly (a lit word is right only if the letters under it are that
word), and the start must be **verified**: a candidate (three words in a row, then two, then one long word) is
accepted only if the next words of the sentence follow it on the page ("et al. 2019" and "compared to the" also
recur). Lit words right on 4 papers: old method 92.9-96.2 %, now **99.3-100 %**; sentences lit in the wrong place
61 → 6, all in formula-dense lines where Docling's text and the text layer really differ. The viewer also got a
**sentence highlight** (now the default), word, or off (`H`).

Found on the way: **the server never used the narration planner** (section 22): pauses, phrase breaks and the
slower pace for maths only reached `pdf-read`, while the viewer and extension - what people use - read flat.
Now `Session.voice_unit` gives each sentence its planned text, pace and pause; the pause is part of the clip.
`bench/test_server.py` checks the aligned timings and the pause: 15/15 pass.

---

## 25. "Task Manager shows Python at 1.5 GB" (while the voice model is < 200 MB)

The user's running server: 139 MB in RAM when idle, but **peak 2.1 GB**, 2.0 GB committed. Measured where it goes:
- indexing page after page: **no leak** - flat at ~760 MB from the first formula on (`bench/memory_over_pages.py`)
- loading the formula model briefly needs ~290 MB extra (ONNX Runtime preparing 334 MB of weights)
- all three document models + the voice: ~1.0 GB; with a page being prepared while the voice speaks, peak
  1.38 GB in a replayed session (`bench/memory_listening.py --fresh`) - the 1.5 GB of the report.

Not a leak: the document models (layout, formulas, tables, OCR) were simply **kept for the whole listening
session**, though they are only needed while pages are prepared - the server indexes the whole paper in a minute
or two, then only the voice works. → `Engine` releases them (and the Node speech process) after 30 s without
indexing and loads them again in ~2 s when a page is needed. Audio cache limited by size (48 MB, minutes of
speech) instead of 300 clips. Replayed session, MTL-LoRA from scratch: **~1.1 GB while the paper is prepared
(first ~1 min), then 440-610 MB for the rest of the listening** (was ~1.0-1.1 GB throughout).

---

## 26. A crash found while measuring delays: docling_parse and a 4,140-path scatter plot

Measuring the listener's waits (`bench/latency_listening.py`: open a paper with an empty cache, play the sentences
in real time) the server died with **heap corruption** (Windows `0xc0000374`) before the first sound - in 2 of 3
runs. Steps, each measured in fresh processes:
- Thread dumps (`python -X faulthandler`): always inside `docling_parse`, two of Docling's threads in it at once.
- Our own pdfium calls (text-layer check, table row repair) ran outside Docling's pdfium lock and left page objects
  to the garbage collector, which can run on any thread → `pdfium_safe.py` (lock + explicit close). Correct, but
  not the cause: still crashing.
- `bench/crash_first_page.py`: 3/10 crashed with the voice loaded, **5/10 without** - not a voice problem.
  Docling's minimal pipeline, none of our code: 8/14 → **a docling_parse 7.22 bug**.
- What did not help: one decoding thread (2/16), every Python call into it under one lock (4/20), single-threaded
  warm-ups (6/16, 12/16). Docling's pdfium backend: 0/14 but reads worse (30/154 spoken units differ: table
  captions lost, "3.12" split, "R d"). docling_parse text + pdfium images: 0/20, but 7/154 differ (two tables lose
  their structure: the layout and table models react to pdfium's different pixels). On-demand rendering with
  docling_parse's own renderer: glyphs missing in parse-only mode (formulas read as "white medium square").
- A worker process restarted on a crash kept the server alive (10/10 runs), but cost a 10-15 s restart each time,
  and the crash came back after a successful warm-up - so not "first conversion only".
- **Per page:** MTL-LoRA page 1 crashed 2/8; its pages 2 and 4, and page 1 of two other papers: 0/8 each. Page 1
  holds a t-SNE scatter plot of **4,140 vector paths**; across 298 pages of seven papers the next busiest had 616,
  the median 1. The renderer races on very large vector drawings - which is why the test papers never crashed.

**Fix:** before converting a page, count its paths (pdfium, milliseconds). Pages above 1,500 (1 of 298) are
converted with docling_parse text + pdfium images (`docling_parse_safe.py`); all others exactly as before, on the
same converter (Docling caches pipelines by options, not backend, so no model loads twice). Results: the crashing
page **0/16**; ordinary pages **154/154 spoken units identical**; MTL page 1 itself **21/21 sentences identical**.
The worker process was dropped again (server ready 25 s instead of 14 s, 775 MB instead of 589 MB).
Lesson: **find the trigger before choosing the fix** - every general workaround changed the output; the specific
one changes nothing.

---

## 27. Like a product: no server to start, same tab, Ctrl+click, models from one place

- **Models:** every model now comes from the project's Hugging Face repository (Vibrant8849/PaperVoice) and is
  loaded from `models/` only - also the layout model (was: Docling's repository, latest version) and the OCR
  models (was: RapidOCR's own download) - each checked against its SHA-256; downloaded by the server itself on
  first start. Not the voice: its training data (Blizzard 2013 Lessac) is research-only and may not be
  redistributed, so it comes from Piper's repository, pinned and checked. Spoken output unchanged (154/154).
- **Loaded on first use only:** OCR (only for pages without a text layer), formula and table models, and now the
  Node maths process (was started with the server: 54 MB for papers without display maths).
- **No server to start:** `pdf-reader-setup` registers a native-messaging helper (current user, Chrome/Edge/Brave);
  the extension asks it to start the server in the background (detached, no window) when it is not answering.
  The server stops after 20 minutes without a request (`--idle-exit`); an open reader tab pings it every minute.
  The extension's ID is fixed by the public key in its manifest, so only it may use the helper. Tested: start
  from the helper (ready 36 s later, survives the helper), no second copy, idle stop after 32 s with a 0.5-minute
  limit, the helper's replies through its `.bat` launcher with no stray bytes.
- **Same tab:** the reader replaces the browser's PDF view in the tab (Back returns to it).
- **Ctrl+click** reads from a sentence; a plain click or drag selects and copies text as on any page.
- **Same tab** was tried and reverted at the user's request: the reader opens in a new tab again. The browser's
  own PDF view cannot be used (extensions cannot read or mark it), so the viewer got real zoom instead
  (Ctrl+wheel, Ctrl +/-/0; pages redrawn at each zoom, checked in headless Edge at 1.5x pixel density).

**"Ctrl+click on a new paragraph takes time"** - measured, a jump to page 9 of a just-opened paper: 24.5 s before
the voice. Causes and fixes:
- the reading list (`Session.sequence`) waited for every page its 20 sentences spanned (it prepared pages 10-13
  first, 16.8 s) → it returns what is prepared and the player asks again as it reads (checked: reading on in
  steps gives exactly the full list, 184/184); it waits only when nothing else is ready;
- the clicked page was not prepared yet → the viewer tells the server which page is in view when scrolling stops,
  and that page is prepared next (finding the sentence: 6.7 s → 0.02 s after looking at the page);
- the audio and word-timing requests for the same sentence each synthesised it, the second behind the first and
  its look-ahead (2.9 s and 6.8 s) → one synthesis, both wait for it;
- wetext (number normalisation) took 97 % of turning a page into sentences, ~1.8 ms per word of each sentence
  with a number → long sentences are normalised in windows of 3 words around each number/symbol: 2.1x faster on
  1,362 sentences of 6 papers, 1,355 identical, the 7 others wetext errors avoided (`Th` read as "Thursday",
  "PDE's", "CI", "EQ") (`bench/check_normalize_windows.py`).
Result, cold start, look at page 6 then Ctrl+click: **24.5 s → 3.4 s** (the rest is synthesising the first
sentence while the background preparation shares the CPU). Copy: checked in headless Edge, selection and Ctrl+C
work (79 characters, copy event, no reading started).

## 28. Highlight and click, measured on every sentence

The user's report: a Ctrl+click sometimes reads another sentence; while reading, the shading sometimes jumps to a
sentence that is not being read; sometimes the blue paragraph box shows instead. Fixing these one at a time had
already caused a regression (DoRA p1: the page-break fix made the ICML notice the click target), so this time the
viewer was **measured first**, on every sentence, in the real viewer:
- `bench/check_viewer.mjs` (headless Edge): for every text sentence in reading order, the viewer's own
  `highlight()` - is every lit word over its own letters, is the band (first to last lit word) the sentence's, does
  any letter get lit by two sentences (right letters in the wrong place), is the blue box shown - and the viewer's
  own click code (`clickContext` → `locateQuery` → `/locate`) at the middle of one of its words.
- `bench/check_playback.mjs`: real Ctrl+clicks (mouse events) and real playback; four times a second the letters
  of the text layer **under the shading on screen** must be the spoken sentence's.

Before (DoRA / MTL-LoRA, pages 1-6): lit correctly 157/182 and 162/174, blue box 9 and 7, **Ctrl+click reads the
clicked sentence 97/173 and 77/167**. Causes, each general, each fixed at its cause:
1. **A click was matched by its line only.** A line usually ends one sentence and starts the next; the server took
   the first sentence containing the line. → The viewer sends the clicked character's offset and the lines
   around it; the server finds that stretch in the paragraph's sentences joined together and the offset picks the
   sentence (`Session._sentence_at`). A caret position is a gap between characters - on a character's right half
   it is the gap after it - so the character whose box holds the point is taken; a click on a "." counts for the
   sentence it ends.
2. **A paragraph continued in the next column has two boxes; only the first was kept.** Its sentences in the second
   column were "not found" → the blue box. → The index keeps all of a block's boxes on the page (`Block.boxes`,
   old caches still load); the viewer searches them all, clicks hit them all.
3. **Symbols broke the anchoring.** A sentence's place is accepted when its next 8 words follow one another; words
   that are only symbols (`||`, `·`) are skipped by that check but were counted in the 8 needed - so any sentence
   with a symbol near its start could never be anchored at its first words. → only words with letters are counted;
   up to two one- or two-letter words (maths letters, sub/superscripts printed elsewhere) may be missing.
4. **A word not found lit the next letters anyway** and pushed every later word along (the drift onto other
   sentences). → it stays unlit and pushes nothing.
5. **The end of a page-break sentence was searched for anywhere on the next page** - its last words ("et al.,
   2019).") were found inside a later sentence and lit there. The letter checks could not see it (the letters were
   right); the overlap check did. → the part on the next page is placed only at the start of its paragraph (what
   joining means), including a word hyphenated across the break ("mag-" / "nitude"); if it cannot be placed it stays
   unlit while the first part is lit (no blue box, and `showSentence`/`tick` still light the first part).
6. A sentence of only tiny words ("0 C, 5°, -5°") had no anchor → looked for whole.
Also: notices printed as body text at the bottom of a page ("Proceedings of the…", "Preprint. Under review.",
"Copyright 2024 by the author(s).") are skipped like footnotes; a click where a removed page-break block was starts
that sentence; the server reports its code version and the extension replaces an outdated one (the user's tab had
kept a stale server alive); **Stop the reading engine** in the popup (`POST /api/shutdown`, local callers only:
another site and a rebinding Host are refused - checked).

After, 12 papers - the 5 worked on and 7 **never looked at before the test** (WaveNet, FastSpeech, two text
normalisation papers, three maths papers):

| | sentences | lit correctly | over another sentence | blue box | Ctrl+click right |
|---|---|---|---|---|---|
| 11 papers with text | 1,290 | **1,290** | 0 | 0 | **1,290 / 1,290** |
| p5 (a scan: no text layer) | 15 | - | - | 15 | - |

Real playback (DoRA, FastSpeech 2): 25/25 real Ctrl+clicks read the clicked sentence (all page-break sentences
included); 882 samples while reading - shading on another sentence 0, nothing shaded 0, blue box 0.
`bench/test_server.py`: all 15 checks pass (487 MB, no torch). A scan keeps the paragraph box: there is no text in
the browser to light (README, known limitations); word boxes from the OCR are a v2 idea. `INDEX_VERSION` 11.


## 29. Named PaperVoice everywhere; a one-line installer

Before the first push (renaming later would break every install): the commands are `papervoice-server`,
`-setup`, `-models`, `-read`, `-convert` (was `pdf-reader`), `-script` (was `pdf-speech`), `-stream`; the project
is `papervoice` (the Python package keeps its name, `pdf_reader`: `[tool.uv.build-backend] module-name`); the
environment variables are `PAPERVOICE_*`; the helper's files moved to `%LOCALAPPDATA%\PaperVoice` (setup removes the
old `pdf-reader` folder; re-run `papervoice-setup` once). Earlier sections keep the names they used.

`install.ps1` (Windows, `irm …/install.ps1 | iex`): installs uv and Node.js when missing (official installer,
winget), gets the code (this copy, else git clone / zip to `%USERPROFILE%\PaperVoice`), `uv sync --locked`,
`npm ci` (tested versions; `npm.cmd`, as `npm.ps1` is blocked by PowerShell's default policy), downloads the models,
registers the helper, opens the extensions page with the folder path on the clipboard. Checked: parses in Windows
PowerShell 5.1; **a fresh copy of exactly the 93 files git would push** installs in 51 s and passes
`bench/test_server.py` (15/15, 572 MB, no torch) - nothing the reader needs is left out by `.gitignore`. The
GitHub repository is `Gowravsharma/PaperVoice` (README, install.ps1, the model card); the models stay on Hugging
Face (`Vibrant8849/PaperVoice`).
---

## Before / after

| | Before | After |
|---|---|---|
| Formula model | CodeFormulaV2, ~300M params, 631 MB | PP-FormulaNet_plus-S, ~58M params, ONNX |
| Time per formula (CPU) | 35–70 s | ~1.4 s |
| 46-page paper | ~3.5 h | ~7 min |
| Quality (same 49 formulas) | 0.939 | 0.927 |
| Runtime deps added | — | `onnxruntime`, `ftfy` |
| PyTorch | needed (layout, tables, formulas) | **not installed** (v1.0, section 23) |
| Whole reading server in memory | ~3 GB | **0.4-0.7 GB** (models load on first use) |

---

## Lessons

- **Measure before tuning.** Profiling showed tuning could never be enough; that saved effort.
- **One run is not a measurement.** The same config varied 2× between runs on this laptop.
- **Check your test data.** A "Print to PDF" copy silently lost the formula text layer.
- **Build an unbiased answer key.** Aligning from the PDF text layer, not from model output, kept the
  comparison fair.
- **Normalise before you score.** Many "errors" were just different spellings of the same LaTeX.
- **Isolate awkward dependencies.** Separate venvs for benchmarking and for exporting avoided breaking the
  main project.
- **Not every optimisation helps.** int8 was slower and worse here; we measured and dropped it.
- **Verify ports bit-for-bit.** The numpy port was checked against the original on every crop.
- **Make every skip visible.** The two worst misses (27 dropped formulas, p5's missing paragraphs) were
  both silent skips; counters (`unreadable_formulas`, `empty_text_blocks`) now report them.
- **Test every input you have.** p5.pdf was the only file without a text layer and the one never tested.
- **Compare each exported graph alone, then the whole output.** Dropout left in an export changed 4 of 14
  tables; per-graph comparison found it at once, and the spoken-text diff proved the final build equivalent.
- **Look for what upstream already ships.** Docling had an ONNX layout engine and an official Heron export;
  only two small pieces were missing.

---

## Open issues / next steps

1. **Speech wording, remaining:** a letter superscript that really is a power (`x^n`) is now read neutrally as
   *"superscript n"*: never wrong, sometimes less natural. Words the model misread beyond repair (`{n d}` for
   "and") need the PDF text layer.
2. **Layout model:** decided by v1.0 - Heron, the only Docling layout model with an ONNX export (Egret-m,
   section 9, would need its own export). A text-layer check (math fonts inside a "text" box) could catch
   equations merged into text.
3. **Long formulas** (100+ tokens) are the weak spot; route them or accept slower/bigger models for them.
4. **Layout mistakes** (text lines boxed as formulas) reach the formula model; filter with the PDF text layer.
5. **Audio:** works end to end with Piper (step 14). The formula and table models now load on first use and
   the server is ready in 8-15 s (section 23). Open choice: the author/affiliation block is read.
6. **Model files** are published as a release archive once the user uploads it (`docs/RELEASING.md`);
   `pdf-reader-models` downloads them.
7. **Tables, v2 ideas:** expand symbols from the caption ("σ" → "standard deviation"); repair broken side-by-side
   parts from the text layer instead of skipping them; say which value is best when the paper bolds it.
   `table_speech.py` is not yet called by `pdf-reader`; it is meant for the full-document script.
8. **After v1.0:** feedback from the first users; the extension is only tested by the user in Edge so far;
   macOS/Linux untested; meaning-from-context reading (v2).

## Where things are

| Path | What |
|---|---|
| `bench/profile_pipeline.py` | per-stage Docling timing for several configs |
| `bench/extract_crops.py` | formula crops (+ optional CodeFormula run) |
| `bench/build_truth.py` | aligns crops to arXiv LaTeX → `refs/fahim_ground_truth.json` |
| `bench/bench_paddle.py`, `bench/bench_onnx.py` | model benchmarks |
| `bench/compare.py` | CodeFormula vs ONNX on identical crops |
| `bench/eval_layout.py` | compares layout models against the arXiv source |
| `bench/measure_resources.py` | wall time, peak RAM and CPU of one pipeline run |
| `src/pdf_reader/formula_onnx.py` | ONNX formula model inside Docling |
| `src/pdf_reader/latex_cleanup.py` | LaTeX tidy-up for speech (step 7) |
| `speech/tex2speech.mjs` | LaTeX → spoken words (`npm install` in `speech/`) |
| `speech/readable.mjs` | paper-notation rewrite before speech (step 8) |
| `src/pdf_reader/table_speech.py` | tables → spoken sentences, with grid repairs (step 10) |
| `src/pdf_reader/script.py` | whole paper → one TTS script (`pdf-speech`, `pdf-reader --speech`) (step 11) |
| `src/pdf_reader/text_norm.py` | citations/links/abbreviations, symbol words, wetext normalisation (step 12) |
| `bench/audit_script.py` | counts everything in a script that is not yet plain words |
| `src/pdf_reader/stream.py` | page-by-page streaming with background prefetch (`pdf-stream`) (step 13) |
| `src/pdf_reader/read_aloud.py` | streamed pages → Piper voice → speakers / WAV (`pdf-read`, `--speed`) (step 14) |
| `src/pdf_reader/index.py` | per-page index: blocks, positions, printed + spoken sentences, click lookup (step 17) |
| `src/pdf_reader/server.py` | local server for the browser viewer (`pdf-reader-server`) (step 17) |
| `viewer/` | the read-along page: `index.html`, `viewer.js`, `viewer.css`; PDF.js via `npm install` (step 18) |
| `src/pdf_reader/timing.py` | word timings from the voice's phonemes, spoken → printed word alignment (step 19) |
| `out/cache/<id>/` | indexed pages per PDF (id = hash of the file), reused on reopen |
| `models/` | model files (`uv run pdf-reader-models`; `PDF_READER_MODELS` to put them elsewhere) |
| `speech/errors.mjs` | prints the MathJax parse error for each failing formula (debugging aid) |
| `speech/compare_reading.mjs` | literal vs readable reading, side by side |
| `out/bench/`, `out/fast/` | benchmark results, full-paper output |
| `src/pdf_reader/onnx_layout.py`, `cpu_only.py` | Heron layout on ONNX without torch/transformers (section 23) |
| `src/pdf_reader/onnx_tableformer.py` | TableFormer on ONNX: decoding loop + Docling table stage (section 23) |
| `src/pdf_reader/vendor/` | torch-free copies of docling-ibm-models' table post-processing (MIT) + numpy helpers |
| `src/pdf_reader/ort_lean.py` | ONNX Runtime sessions without memory arenas |
| `src/pdf_reader/models.py` | model paths, checksums, `pdf-reader-models` downloader |
| `src/pdf_reader/narration.py`, `prosody.py` | pauses, pace, prosodic phrasing (section 22) |
| `extension/` | Chrome/Edge extension (Manifest V3) |
| `bench/export_tableformer.py`, `bench/vendor_tableformer.py` | make the table ONNX model and the vendored code |
| `bench/check_onnx_layout.py` | ONNX vs PyTorch layout, pixels and boxes |
| `bench/memory_breakdown.py`, `bench/memory_per_model.py` | memory per pipeline step / per model |
| `bench/package_models.py` | release archive of the models + SHA-256 into models.py |
| `bench/test_server.py` | end-to-end server test incl. security checks |
| `bench/check_highlight.py` | replays the viewer's word locating on every sentence of an indexed PDF |
| `bench/check_viewer.mjs`, `bench/check_playback.mjs` | every sentence through the real viewer; real clicks + playback (section 28) |
| `src/pdf_reader/docling_parse_safe.py`, `pdfium_safe.py` | crash-safe PDF handling (section 26) |
| `bench/crash_first_page.py`, `bench/latency_listening.py` | crash rate of fresh processes; waits a listener hears |
| `bench/audit_pipeline.py`, `make_samples.py`, `speak_tables.py` | per-stage audit, listening samples |
| `docs/RELEASING.md` | how to publish a version (model archive, release URL) |
