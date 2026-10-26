"""
stream.py - page-by-page streaming: the first page is ready to read in seconds while the rest is prepared in
the background, like a game streaming the chunk around the player instead of loading the whole map.

    streamer = PageStreamer("paper.pdf", prefetch=3)
    page = streamer.get(1)      # blocks only until page 1 is ready
    ... speak page.text ...     # meanwhile pages 2-4 are being prepared
    page = streamer.get(2)      # usually ready already; get(20) jumps ahead (re-prioritises)

Why it works: preparing a page takes ~2-7 s on this CPU; listening to it takes ~3-4 minutes, so the worker
stays far ahead of the listener.

Per page: Docling (layout, formulas via ONNX, tables) on that page only -> index.page_blocks (blocks in reading
order, each with its position and its sentences, printed and spoken).
Carried between pages:
  - "inside the References section" (skipped until the next heading)
  - a sentence cut by the page break: page p keeps it as its `tail`; if page p+1 is prepared right after p, the
    tail is joined to p+1's first paragraph and marked joined. A reader finishing page p says the tail only if it
    was not joined, so it is never lost (jumps, out-of-order indexing) and never said twice.

Engine: the loaded models (layout, formula model, Node speech process, wetext; OCR on first need), shared by all
streamers in one process (the server opens several PDFs without reloading). index_all=True keeps preparing the
whole document outward from the listener's page; cache_dir saves each finished page so reopening is instant.

CLI (measures time to first page and whether a listener would ever wait):
    uv run papervoice-stream paper.pdf [--pages 2,5,9-11 | --start 12] [--prefetch 3]
                     [--wpm 150] [--out out/stream]
"""
import argparse
import gc
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from docling.datamodel.base_models import InputFormat

from pdf_reader.docling_parse_safe import HEAVY_PATHS, PdfiumImageDocumentBackend
from pdf_reader.docling_parse_safe import backend_options as safe_backend_options
from pdf_reader.formula_onnx import make_converter
from pdf_reader.index import Sentence, blocks_text, from_json, join_tail, page_blocks, take_tail, to_json
from pdf_reader.pdfium_safe import char_count, open_document, page_count, path_count
from pdf_reader.script import FormulaSpeaker, new_stats
from pdf_reader.text_norm import normalize

INDEX_VERSION = 11  # bump when the index format or the reading rules change, so old caches are rebuilt
# 2: "inside References" is taken from the page before, not from whatever page was indexed last
# 3: a page indexed on a wrong guess about "inside References" is re-indexed when the page before is done


@dataclass
class Page:
    number: int
    blocks: list  # index.Block, reading order
    ready_at: float  # seconds since the streamer started
    seconds: float  # time spent preparing this page
    stats: dict = field(default_factory=dict)
    tail: dict = None  # unfinished last sentence (see module doc), {"block", "sentence"}
    tail_joined: bool = False
    in_refs_after: bool = False  # still inside the References section at the end of this page
    in_refs_before: bool = False  # what was assumed at the start (a guess if the page before was not ready)

    @property
    def tail_sentence(self):
        return Sentence(**self.tail["sentence"]) if self.tail else None

    @property
    def text(self):
        """The page as a script; includes the tail unless the next page took it over."""
        text = blocks_text(self.blocks)
        if self.tail and not self.tail_joined:
            text += "\n\n" + self.tail["sentence"]["spoken"]
        return text.strip()

    @property
    def words(self):
        return len(self.text.split())

    def to_json(self):
        return {"version": INDEX_VERSION, "number": self.number, "blocks": to_json(self.blocks),
                "stats": self.stats, "tail": self.tail, "tail_joined": self.tail_joined,
                "in_refs_after": self.in_refs_after, "in_refs_before": self.in_refs_before}

    @classmethod
    def from_json(cls, d):
        return cls(d["number"], from_json(d["blocks"]), 0.0, 0.0, d["stats"], d["tail"], d["tail_joined"],
                   d.get("in_refs_after", False), d.get("in_refs_before", False))


class Engine:
    """The loaded models, shared by every PageStreamer in the process. index_page() is serialised by a lock:
    the converters and the Node speech process are shared.

    The document models (layout, formulas, tables, OCR: ~0.4-0.6 GB) are only needed while pages are prepared.
    Once no page has been prepared for IDLE_SECONDS - the whole document is indexed and the listener is just
    listening - they are released, and loaded again (a few seconds) when a page is next needed."""

    IDLE_SECONDS = 30

    def __init__(self, threads=4):
        t = time.perf_counter()
        self.threads = threads
        self.lock = threading.Lock()
        self.conv = self.conv_ocr = self.speaker = None
        self._load()
        normalize("warm up 1")  # wetext loads its grammars on first use (~3 s)
        self.load_seconds = time.perf_counter() - t
        self.last_used = time.monotonic()
        threading.Thread(target=self._release_when_idle, daemon=True).start()

    def _load(self):
        self.conv = make_converter(self.threads)
        self.conv.initialize_pipeline(InputFormat.PDF)
        self.speaker = FormulaSpeaker()
        # conv_ocr is created the first time a page without a text layer is met

    def index_page(self, pdf_path, pdf, page, ocr, state, stats):
        """Convert one page with Docling and index it (index.page_blocks)."""
        with self.lock:
            self.last_used = time.monotonic()
            if self.conv is None:
                self._load()
            if ocr and self.conv_ocr is None:
                self.conv_ocr = make_converter(self.threads, ocr=True)
                self.conv_ocr.initialize_pipeline(InputFormat.PDF)
            conv = self.conv_ocr if ocr else self.conv
            # a page with a huge vector drawing crashes docling_parse's renderer: its images come from pdfium
            # (docling_parse_safe.py); every other page is converted with Docling's default backend
            heavy = path_count(pdf, page) > HEAVY_PATHS
            fopt = conv.format_to_options[InputFormat.PDF]
            default = (fopt.backend, fopt.backend_options)
            if heavy:
                fopt.backend, fopt.backend_options = PdfiumImageDocumentBackend, safe_backend_options()
            try:
                doc = conv.convert(pdf_path, page_range=(page, page)).document
            finally:
                fopt.backend, fopt.backend_options = default
            stats["pdfium_images"] = heavy
            blocks = page_blocks(doc, pdf, self.speaker, state, stats, page)
            self.last_used = time.monotonic()
            return blocks

    def release(self):
        """Drop the document models and the Node speech process (they load again on the next page)."""
        with self.lock:
            if self.conv is None:
                return
            self.conv = self.conv_ocr = None
            self.speaker.close()
            self.speaker = None
            gc.collect()

    def _release_when_idle(self):
        while True:
            time.sleep(5)
            if self.conv is not None and time.monotonic() - self.last_used > self.IDLE_SECONDS:
                self.release()

    def close(self):
        if self.speaker:
            self.speaker.close()


def parse_pages(spec, n_pages, start=None):
    """Pages to read, in order. spec: "5", "3-7", "2,5,9-11", "7-" (to the end); start=N means N to the end.
    Out-of-range pages are dropped; an empty result is an error."""
    if start is not None:
        spec = f"{start}-"
    if not spec:
        return list(range(1, n_pages + 1))
    pages = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages += range(int(lo or 1), (int(hi) if hi else n_pages) + 1)
        elif part:
            pages.append(int(part))
    pages = [p for p in dict.fromkeys(pages) if 1 <= p <= n_pages]
    if not pages:
        raise ValueError(f"no pages of 1-{n_pages} in {spec!r}")
    return pages


class PageStreamer:
    def __init__(self, pdf_path, threads=4, prefetch=3, keep_references=False, pages=None,
                 engine=None, index_all=False, cache_dir=None):
        """pages: the pages to read, in order (default: all); the worker prepares only these, starting
        with the first, so `--start 20` does not spend time on pages 1-19.
        engine: shared loaded models (created in the background if None).
        index_all: after the pages ahead of the listener, keep going through the rest of the document.
        cache_dir: load finished pages from here and save new ones (keyed by the caller, e.g. a file hash)."""
        self.pdf_path = str(pdf_path)
        self.pdf = open_document(self.pdf_path)
        self.n_pages = page_count(self.pdf)
        self.order = list(pages) if pages else list(range(1, self.n_pages + 1))
        self.threads, self.prefetch, self.index_all = threads, prefetch, index_all
        self.state = {"keep_references": keep_references, "in_refs": False}
        self.cursor = self.order[0]  # page the listener wants now; the worker starts preparing here
        self.pages = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self._load_cache()
        self.cond = threading.Condition()
        self.t0 = time.perf_counter()
        self.engine, self.own_engine = engine, engine is None
        self.load_seconds = 0.0 if engine else None
        self.error = None
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # ------------------------------------------------------------ cache
    def _cache_file(self, p):
        return self.cache_dir / f"page_{p:04d}.json"

    def _load_cache(self):
        for p in range(1, self.n_pages + 1):
            f = self._cache_file(p)
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                if d.get("version") == INDEX_VERSION:
                    self.pages[p] = Page.from_json(d)

    def _save(self, *pages):
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for pg in pages:
                self._cache_file(pg.number).write_text(json.dumps(pg.to_json(), ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ listener side
    def get(self, number, priority=True):
        """Blocks until the page is ready. priority=True: this is where the listener is, so it becomes the
        worker's next job. Internal look-ups (prefetching audio, the page-break check) pass priority=False and
        just wait - otherwise they would keep pulling the worker back and a jump would starve."""
        with self.cond:
            if priority:
                self.cursor = number
            self.cond.notify_all()
            while number not in self.pages and self.error is None:
                self.cond.wait(0.5)
            if self.error:
                raise RuntimeError(f"page worker failed: {self.error}")
            return self.pages[number]

    def ready(self, number):
        with self.cond:
            return self.pages.get(number)

    def focus(self, number):
        """The listener is now on this page: index it and the pages after it first. Does not wait."""
        with self.cond:
            if self.cursor != number:
                self.cursor = number
                self.cond.notify_all()

    def pause(self, paused=True):
        """Stop background indexing (another document is being read); pages asked for with get() are still
        prepared. pause(False) resumes."""
        with self.cond:
            self.paused = paused
            self.cond.notify_all()

    # ------------------------------------------------------------ worker side
    def _next_page(self):
        """First unprepared page in reading order from the cursor, at most `prefetch` pages ahead; with
        index_all, then any page left (after the cursor first, then from the start). None when nothing to do.
        A cursor outside the list (a jump) is honoured on its own."""
        if self.cursor not in self.order:
            return None if self.cursor in self.pages else self.cursor
        if getattr(self, "paused", False):  # background indexing paused: only the page being asked for
            return None if self.cursor in self.pages else self.cursor
        i = self.order.index(self.cursor)
        ahead = self.order[i:] if self.index_all else self.order[i:i + self.prefetch + 1]
        rest = self.order[:i] if self.index_all else []
        for p in ahead + rest:
            if p not in self.pages:
                return p
        return None

    def _run(self):
        try:
            if self.engine is None:
                self.engine = Engine(self.threads)
                self.load_seconds = self.engine.load_seconds
            while True:
                with self.cond:
                    p = self._next_page()
                    while p is None:
                        if all(q in self.pages for q in self.order):
                            return
                        self.cond.wait()
                        p = self._next_page()
                self._prepare(p)
        except Exception as e:  # surface it to the listener instead of hanging
            with self.cond:
                self.error = repr(e)
                self.cond.notify_all()
        finally:
            if self.own_engine and self.engine:
                self.engine.close()

    def has_text_layer(self, p):
        """False for scans and "Print to PDF" copies whose letters are vector outlines (p5.pdf: 0 characters,
        1,519 paths). Such pages need OCR, or their paragraphs come out empty."""
        return char_count(self.pdf, p) >= 20

    def _prepare(self, p):
        t = time.perf_counter()
        prev = self.pages.get(p - 1)
        # whether we are inside the References section comes from the page just before (pages may be indexed
        # out of order: after the last page, "in References" must not leak into page 2)
        self.state["in_refs"] = prev.in_refs_after if prev is not None else False
        in_refs_before = self.state["in_refs"]
        ocr = not self.has_text_layer(p)
        stats = new_stats()
        stats["ocr"] = ocr
        blocks = self.engine.index_page(self.pdf_path, self.pdf, p, ocr, self.state, stats)
        if prev is not None and prev.tail and not prev.tail_joined and join_tail(prev.tail, blocks):
            prev.tail_joined = True
        tail = take_tail(blocks) if p < self.n_pages and (p + 1) not in self.pages else None
        page = Page(p, blocks, time.perf_counter() - self.t0, time.perf_counter() - t, stats, tail,
                    in_refs_after=self.state["in_refs"], in_refs_before=in_refs_before)
        with self.cond:
            self.pages[p] = page
            # the next page was indexed earlier (a jump) with a guess about "inside References"; if this page
            # proves the guess wrong, index it again (it then corrects the pages after it the same way)
            nxt = self.pages.get(p + 1)
            if nxt is not None and nxt.in_refs_before != page.in_refs_after:
                del self.pages[p + 1]
                if self.cache_dir:
                    self._cache_file(p + 1).unlink(missing_ok=True)
            self.cond.notify_all()
        self._save(page, *([prev] if prev is not None and prev.tail_joined else []))


def simulate(pages, wpm):
    """Playback timeline if a listener starts as soon as page 1 is ready: (first audio at, stalls, total stall)."""
    clock, stalls, waited = None, 0, 0.0
    for pg in pages:
        if clock is None:
            clock = pg.ready_at
        elif pg.ready_at > clock:
            stalls += 1
            waited += pg.ready_at - clock
            clock = pg.ready_at
        clock += pg.words / wpm * 60
    return pages[0].ready_at, stalls, waited, clock


def main():
    ap = argparse.ArgumentParser(description="Stream a PDF page by page into TTS-ready text")
    ap.add_argument("pdf")
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wpm", type=float, default=150, help="speaking rate for the playback simulation")
    ap.add_argument("--out", default="out/stream")
    ap.add_argument("--pages", help='pages to prepare: "5", "3-7", "2,5,9-11", "7-" (default: all)')
    ap.add_argument("--start", type=int, help="start at this page and continue to the end")
    a = ap.parse_args()

    # measurement mode: the "listener" asks for each page in turn as soon as the previous one is ready, so the
    # worker runs flat out; the playback timeline is then computed from the ready times at --wpm
    wanted = parse_pages(a.pages, page_count(open_document(a.pdf)), a.start)
    s = PageStreamer(a.pdf, a.threads, prefetch=10**6, pages=wanted)
    pages = []
    for p in wanted:
        pg = s.get(p)
        pages.append(pg)
        print(f"page {p:3d} ready at {pg.ready_at:6.1f}s  (prepared in {pg.seconds:5.1f}s, {pg.words:5d} words, "
              f"{pg.stats['formulas']} formulas, {pg.stats['tables']} tables"
              + (", OCR: no text layer" if pg.stats.get("ocr") else "")
              + (f", {pg.stats['unreadable_formulas']} unreadable formulas" if pg.stats["unreadable_formulas"] else "")
              + (f", WARNING {pg.stats['empty_text_blocks']} empty text blocks" if pg.stats["empty_text_blocks"] else "")
              + ")", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.pdf).stem
    (out / f"{stem}.speech.txt").write_text("\n\n".join(pg.text for pg in pages if pg.text) + "\n", encoding="utf-8")
    first, stalls, waited, end = simulate(pages, a.wpm)
    total = sum(pg.words for pg in pages)
    print(f"\nmodels loaded in {s.load_seconds:.1f}s; {len(pages)} of {s.n_pages} pages prepared in "
          f"{pages[-1].ready_at:.0f}s")
    print(f"first page ready after {first:.1f}s; listening at {a.wpm:.0f} words/min takes "
          f"{total / a.wpm:.0f} min; listener waits: {stalls} times, {waited:.1f}s in total")
    print(f"wrote {out / stem}.speech.txt ({total} words)")
