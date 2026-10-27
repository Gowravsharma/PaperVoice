"""
script.py - turn a converted paper (Docling JSON) into one plain-text script for a TTS engine.

Usage:
    uv run papervoice-script out/paper.json --pdf paper.pdf [--out out/paper.speech.txt] [--keep-references]
    (or `papervoice-convert paper.pdf --speech` to convert and write the script in one go)

Reading order is Docling's. One paragraph per line, blank line between paragraphs (TTS engines pause there).
  headings        own paragraph; ALL-CAPS headings are put in title case so they are read as words
  text, lists     text_norm.prose (citations dropped, "(2.1)" -> "equation 2.1", links, abbreviations) and
                  text_norm.symbols; inline math is read as the PDF has it (v1)
  formulas        LaTeX -> words with speech/tex2speech.mjs (paper-notation rules), one Node call per document
  tables          table_speech.speak_table; caption paragraphs it reads are not read again
  figures         image skipped, caption read
  skipped         page headers/footers (Docling furniture), footnotes, and the reference list (from a
                  "References" heading to the next heading, so an appendix after it is still read)
Finally every paragraph goes through text_norm.normalize (wetext): numbers, years, dates, ordinals -> words.
"""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem

from pdf_reader.latex_cleanup import tidy
from pdf_reader.table_speech import speak_table
from pdf_reader import text_norm
from pdf_reader.text_norm import normalize, prose, symbols

ROOT = Path(__file__).resolve().parents[2]
TEX2SPEECH = ROOT / "speech" / "tex2speech.mjs"
PRIVATE_NODE = ROOT / "tools" / "node" / "node.exe"  # the copy install.ps1 downloads when Node.js is not installed


def node_exe():
    """Node.js for the maths speech: the one on the PATH, else the private copy install.ps1 put in tools/."""
    return shutil.which("node") or (str(PRIVATE_NODE) if PRIVATE_NODE.exists() else "node")
REFERENCES = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\.?\s*$", re.I)
EQ_NUMBER = re.compile(r"\(\s*\d+(?:\.\d+)*\s*\)")
SKIP = {DocItemLabel.FOOTNOTE, DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.PICTURE}
# a venue or copyright notice the layout model labelled as body text (DoRA, ICML 2024: "Proceedings of the 41st
# International Conference on Machine Learning, ... Copyright 2024 by the author(s).", bottom of column 1):
# skipped like a footnote when it is also in the bottom part of its page
NOTICE = re.compile(r"^\W*(Proceedings of the\b|Preprint\b|Under review\b|Accepted (at|to|for|by)\b|"
                    r"Published (at|in|as)\b|Equal contribution|Correspondence to\b)|"
                    r"Copyright (©\s*)?\d{4} by the author", re.I)


def is_notice(it, doc):
    text = getattr(it, "text", "") or ""
    if not NOTICE.search(text) or not it.prov:
        return False
    prov = it.prov[0]
    page = doc.pages.get(prov.page_no)
    if page is None:
        return False
    top = prov.bbox.to_top_left_origin(page.size.height).t
    return top > 0.75 * page.size.height


UNREADABLE = "A formula that could not be read."


class FormulaSpeaker:
    """A long-running `tex2speech.mjs --serve` process: Node and the speech rules load once (~0.8 s),
    then each formula is one line in, one line out. Started on the first formula, so a paper without display
    maths never runs it (~55 MB)."""

    def __init__(self):
        self.proc = None

    def __call__(self, latex_list):
        out = []
        for latex in latex_list:
            if not latex.strip():  # nothing recognised; "$?$" would be read as "question mark"
                out.append(UNREADABLE)
                continue
            if self.proc is None:
                # no console window: started from the background server, Windows would open a terminal for it
                self.proc = subprocess.Popen([node_exe(), str(TEX2SPEECH), "clearspeak", "--serve"],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                             encoding="utf-8", bufsize=1,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.proc.stdin.write("$" + " ".join(latex.split()) + "$\n")
            self.proc.stdin.flush()
            s = self.proc.stdout.readline().rstrip("\n")
            # MathJax could not parse it -> the raw LaTeX comes back; never send LaTeX to a TTS engine
            out.append(s if s and "\\" not in s and "$" not in s else UNREADABLE)
        return out

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
            self.proc = None


SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}


def heading(text):
    text = " ".join(text.split())
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        words = []
        for i, w in enumerate(text.split()):
            if w.lower() in SMALL_WORDS and i:
                words.append(w.lower())
            elif re.fullmatch(r"[\dA-Z]{1,3}\.?\d*:?", w):  # numbers and short acronyms: 3.2, TTS, A
                words.append(w)
            else:
                words.append(w.capitalize())
        text = " ".join(words)
    return symbols(text).rstrip(".:,;") + "."


def script_parts(doc: DoclingDocument, pdf, speaker, state, stats):
    """Items of one document (or one page) -> [(kind, text)] in reading order, not yet normalised.
    kind: heading | text | caption | formula | table. `state` carries over between pages ({"in_refs", "keep_references"});
    `stats` counts are added to."""
    return [(kind, text) for kind, text, _ in script_items(doc, pdf, speaker, state, stats)]


def script_items(doc: DoclingDocument, pdf, speaker, state, stats):
    """Like script_parts, but each part also carries the Docling item it came from (for its page position)."""
    items = [it for it, _ in doc.iterate_items()]
    # a formula box holding only an equation number, e.g. "(3.25)" next to an equation boxed separately:
    # nothing to say, and not an unreadable formula either
    number_only = {id(it) for it in items if it.label == DocItemLabel.FORMULA and not tidy(it.text)[0].strip()
                   and EQ_NUMBER.fullmatch((it.text or it.orig or "").strip())}
    items = [it for it in items if id(it) not in number_only]
    formulas = [it for it in items if it.label == DocItemLabel.FORMULA]
    # tidy again: idempotent, and covers JSON written before a cleanup rule existed
    spoken = dict(zip(map(id, formulas), speaker([tidy(it.text)[0] for it in formulas])))
    stats["unreadable_formulas"] += sum(s == UNREADABLE for s in spoken.values())

    parts, consumed = [], set()
    for it in items:
        if id(it) in consumed or it.label in SKIP:
            continue
        if it.label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM) and is_notice(it, doc):
            stats["skipped_notices"] = stats.get("skipped_notices", 0) + 1
            continue
        text = getattr(it, "text", "") or ""
        if it.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            state["in_refs"] = bool(REFERENCES.match(text)) and not state.get("keep_references")
            if not state["in_refs"]:
                parts.append(("heading", heading(text), it))
            continue
        if state.get("in_refs"):
            stats["skipped_reference_items"] += 1
            continue
        if isinstance(it, TableItem):
            st = speak_table(it, doc, pdf)
            consumed.update(map(id, st.consumed))
            consumed.update(id(c.resolve(doc)) for c in it.captions)
            if st.text:
                parts.append(("table", st.text, it))
                stats["tables"] += 1
        elif it.label == DocItemLabel.FORMULA:
            parts.append(("formula", spoken[id(it)], it))
            stats["formulas"] += 1
        elif text.strip():
            kind = "caption" if it.label == DocItemLabel.CAPTION else "text"
            parts.append((kind, symbols(prose(" ".join(text.split()))), it))
            stats["paragraphs"] += 1
        elif it.label in (DocItemLabel.TEXT, DocItemLabel.LIST_ITEM, DocItemLabel.CAPTION):
            # a text block with no words: the PDF has no text layer there (scan, "Print to PDF" outlines)
            # and OCR was off. Count it so this is never silently skipped again.
            stats["empty_text_blocks"] += 1
    return parts


def new_stats():
    return {"paragraphs": 0, "formulas": 0, "tables": 0, "skipped_reference_items": 0,
            "unreadable_formulas": 0, "normalizer_failures": 0, "empty_text_blocks": 0}


def normalize_parts(parts, stats):
    text_norm.FAILURES.clear()
    out = [text_norm.normalize_numbers(t) if kind == "formula" else normalize(t) for kind, t in parts]
    stats["normalizer_failures"] += len(text_norm.FAILURES)
    return [p for p in out if p.strip()]


def build_script(doc: DoclingDocument, pdf=None, keep_references=False, speaker=None):
    """Whole document -> (script text, stats)."""
    own = speaker is None
    speaker = speaker or FormulaSpeaker()
    try:
        stats = new_stats()
        parts = script_parts(doc, pdf, speaker, {"keep_references": keep_references}, stats)
    finally:
        if own:
            speaker.close()
    return "\n\n".join(normalize_parts(parts, stats)) + "\n", stats


def write_script(json_path, pdf_path=None, out_path=None, keep_references=False):
    doc = DoclingDocument.load_from_json(Path(json_path))
    pdf = None
    if pdf_path:
        from pdf_reader.pdfium_safe import open_document
        pdf = open_document(pdf_path)
    text, stats = build_script(doc, pdf, keep_references)
    out_path = Path(out_path or Path(json_path).with_suffix(".speech.txt"))
    out_path.write_text(text, encoding="utf-8")
    return out_path, text, stats


def main():
    ap = argparse.ArgumentParser(description="Docling JSON -> plain-text script for TTS")
    ap.add_argument("json")
    ap.add_argument("--pdf", help="source PDF (lets the table reader repair rows from the text layer)")
    ap.add_argument("--out")
    ap.add_argument("--keep-references", action="store_true")
    a = ap.parse_args()
    out_path, text, stats = write_script(a.json, a.pdf, a.out, a.keep_references)
    print(f"wrote {out_path}: {len(text.split())} words; {stats}")
