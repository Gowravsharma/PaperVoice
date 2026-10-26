"""
pdfium_safe.py - use pdfium (pypdfium2) safely next to Docling's worker threads.

pdfium is not thread-safe, so Docling runs every pdfium call under its global lock (docling.utils.locks.
pypdfium2_lock). Our own calls (the text-layer check, the table row repair) must take the same lock - and must
close their page objects themselves: pypdfium2 otherwise closes them in a garbage-collection finaliser, which runs
on whichever thread happens to allocate memory (often the voice thread), at a moment when Docling may be inside
pdfium on another thread. That race corrupted the heap and crashed the server (Windows 0xc0000374, seen in
bench/latency_listening.py: a crash in docling_parse with the voice running; never with indexing alone).
"""
from contextlib import contextmanager

import pypdfium2 as pdfium
from docling.utils.locks import pypdfium2_lock


def open_document(path):
    with pypdfium2_lock:
        return pdfium.PdfDocument(str(path))


def page_count(pdf):
    with pypdfium2_lock:
        return len(pdf)


@contextmanager
def text_page(pdf, page_no):
    """(page, text page) of page `page_no` (1-based), used and closed under Docling's pdfium lock."""
    with pypdfium2_lock:
        page = pdf[page_no - 1]
        tp = page.get_textpage()
        try:
            yield page, tp
        finally:
            tp.close()
            page.close()


def char_count(pdf, page_no):
    with text_page(pdf, page_no) as (_, tp):
        return tp.count_chars()


def path_count(pdf, page_no):
    """Vector paths on a page (lines, curves, scatter-plot dots), counted without rendering."""
    with pypdfium2_lock:
        page = pdf[page_no - 1]
        try:
            return sum(1 for o in page.get_objects(max_depth=3) if o.type == pdfium.raw.FPDF_PAGEOBJ_PATH)
        finally:
            page.close()
