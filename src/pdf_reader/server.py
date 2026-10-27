"""
server.py - local companion server for the PDF viewer / browser extension (localhost only).

    uv run papervoice-server [--port 8765]

The browser shows the PDF and highlights; this process does the heavy work (Docling, formula model, voice).
Nothing leaves the machine: it listens on 127.0.0.1 only.

Who may call it: only requests addressed to 127.0.0.1 / localhost on this port (a page on another site that
points a DNS name at 127.0.0.1 is refused), coming from the viewer itself (same origin), from a browser extension,
or from a local program (no Origin header). No CORS headers are sent: other websites cannot read the answers, and
their requests are refused outright. The browser extension (extension/) is allowed by its host permission.

API (JSON unless noted):
  GET  /api/health                               {"ready": models loaded?}
  POST /api/open[?name=paper.pdf]   body: the PDF bytes (Content-Type: application/pdf) or {"path": "C:/..."}
                        -> {"doc", "pages", "name", "indexed"}; indexing starts in the background, outward
                           from page 1 (or from wherever the reader is), and is cached per file content
  GET  /api/doc/<id>/pdf                         the PDF itself (for the viewer, e.g. after the extension opened it)
  GET  /api/doc/<id>/page/<p>[?wait=0]           blocks of page p (makes p the indexer's next job)
  GET  /api/doc/<id>/locate?page=&x=&y=[&text=]  the sentence to start from after a click at (x, y) in PDF
                                                 points (top-left origin); text = the clicked line, to pick the
                                                 right sentence inside a paragraph
  GET  /api/doc/<id>/sequence?from=<sid>&count=20   the next sentences in reading order, across pages
  GET  /api/doc/<id>/audio/<sid>?speed=1.0       WAV of one sentence (audio/wav); the following sentences are
                                                 synthesised in the background so playback never waits
  GET  /api/doc/<id>/timing/<sid>?speed=1.0      when each spoken word is said, and which printed word it is
                                                 (for word-by-word highlighting; see timing.py)

Seeking is a lookup: every page is indexed once (blocks in reading order with positions and sentences), so a
click maps to a sentence in a list of a few dozen blocks; only a page that is not indexed yet costs the page
time (2-7 s, ~15 s with OCR), and it jumps to the front of the queue.
"""
import argparse
import hashlib
import io
import json
import os
import re
import threading
import time
import wave
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pdf_reader.index import Sentence, block_boxes, in_box, locate

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "out" / "cache"
CODE_VERSION = __import__("pdf_reader.native_host", fromlist=["code_version"]).code_version()  # as started
VIEWER = ROOT / "viewer"  # the reading page (index.html, viewer.js, viewer.css)
PDFJS = VIEWER / "node_modules" / "pdfjs-dist" / "build"  # npm install in viewer/
LAST_REQUEST = [time.monotonic()]  # any accepted request (the viewer pings every minute while it is open)
AUDIO_CACHE_BYTES = 48 * 2**20  # ~9 minutes of speech at 22 kHz, 16-bit
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
                ".css": "text/css", ".map": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}


def letters(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Session:
    """One open PDF: its streamer (indexer) and reading-order helpers."""

    def __init__(self, doc_id, pdf_path, engine, name):
        from pdf_reader.stream import PageStreamer
        self.id, self.name = doc_id, name
        self.streamer = PageStreamer(pdf_path, engine=engine, index_all=True, cache_dir=CACHE / doc_id)
        self.n_pages = self.streamer.n_pages

    def page(self, p, wait=True, priority=False):
        """priority=True only for what the user does (open a page, click); internal reads must not move the
        indexer, or a jump would be starved by look-ahead requests."""
        return self.streamer.get(p, priority) if wait else self.streamer.ready(p)

    def page_json(self, pg):
        return {"page": pg.number, "ready": True, "blocks": [self._block_json(b) for b in pg.blocks],
                "tail": pg.tail, "tail_joined": pg.tail_joined, "stats": pg.stats}

    @staticmethod
    def _block_json(b):
        return {"id": b.id, "kind": b.kind, "bbox": b.bbox, "boxes": block_boxes(b), "continues_from": b.continues_from,
                "sentences": [{"id": s.id, "text": s.text, "spoken": s.spoken} for s in b.sentences]}

    def _page_sentences(self, p, wait_next=True):
        """((sentence, block) pairs of page p in reading order, with the page-break tail where it belongs; complete).
        The tail (a sentence cut by the page break) belongs here only if the next page did not take it over, which
        is known once the next page is prepared: with wait_next=False and that page not ready yet, the tail is left
        out and complete is False."""
        pg = self.page(p)
        out = [(s, b) for b in pg.blocks for s in b.sentences]
        if pg.tail:
            if p + 1 <= self.n_pages:
                if not wait_next and self.streamer.ready(p + 1) is None:
                    return out, False
                self.page(p + 1)  # decides whether the next page took the sentence over
            if not pg.tail_joined:
                block_id = pg.tail["block"]
                block = next((b for b in pg.blocks if b.id == block_id), None)
                out.append((Sentence(**pg.tail["sentence"]), block))
        return out, True

    def item(self, sid):
        """One sentence by id, looked up on its own page only. Unlike sequence(), this never waits for the next
        page to be indexed - fetching a sentence's audio must not depend on the page after it."""
        p = int(sid.split(".")[0])
        pg = self.page(p)
        for b in pg.blocks:
            for s in b.sentences:
                if s.id == sid:
                    return {"id": s.id, "page": p, "block": b.id, "kind": b.kind, "bbox": b.bbox,
                            "boxes": block_boxes(b), "continues_from": b.continues_from,
                            "prev_part": self._prev_part(b, s),
                            "text": s.text, "spoken": s.spoken}
        if pg.tail and pg.tail["sentence"]["id"] == sid:
            t = pg.tail["sentence"]
            return {"id": t["id"], "page": p, "block": pg.tail["block"], "kind": "text", "bbox": None, "boxes": [],
                    "continues_from": "", "text": t["text"], "spoken": t["spoken"]}
        return None

    def _prev_part(self, block, sentence):
        """For a sentence that starts on the previous page (the first sentence of a block that continues a
        paragraph from there): where its first part is, {"page", "bbox"}, so the viewer can light and show those
        words too. None otherwise."""
        if block is None or not block.continues_from or not block.sentences or block.sentences[0].id != sentence.id:
            return None
        pp = int(block.continues_from.split(".")[0])
        prev = self.streamer.ready(pp)
        if prev is None:
            return None
        b = next((x for x in prev.blocks if x.id == block.continues_from), None)
        if b is not None:
            return {"page": pp, "bbox": b.bbox, "boxes": block_boxes(b)}
        # the block held only this sentence and is gone from its page: its position was kept with the sentence
        tail = prev.tail if prev.tail and prev.tail.get("block") == block.continues_from else None
        if not tail or not tail.get("bbox"):
            return None
        return {"page": pp, "bbox": tail["bbox"], "boxes": tail.get("boxes") or [tail["bbox"]]}

    def voice_unit(self, sid):
        """How a sentence is voiced: (text with phrase breaks, pace, pause class after it) from the narration
        planner (narration.plan_page), the same as papervoice-read. Planned once per page."""
        from pdf_reader.narration import plan_page
        from pdf_reader.prosody import phrase
        p = int(sid.split(".")[0])
        plans = self.__dict__.setdefault("_plans", {})
        if p not in plans:
            plans[p] = {u.sentence: (u.text, u.pace, u.pause) for u in plan_page(self.page(p).blocks, p)}
        if sid in plans[p]:
            return plans[p][sid]
        item = self.item(sid)  # the page-break tail: a sentence fragment read at the end of its page
        text, pace = phrase(item["spoken"])
        return text, pace, "sentence"

    def sequence(self, start_id, count):
        """Up to `count` sentences in reading order from start_id. Returns what is prepared already instead of waiting
        for later pages (the player asks again as it reads on); it waits only when nothing beyond start_id would be
        returned. A jump to a page far ahead used to prepare 4-5 pages before the first word (24.5 s)."""
        p = int(start_id.split(".")[0])
        items, started = [], False
        while p <= self.n_pages and len(items) < count:
            if len(items) > 1 and self.streamer.ready(p) is None:
                break  # this page is still being prepared: the player asks again later
            pairs, complete = self._page_sentences(p, wait_next=False)
            if not complete and len(items) + self._count_from(pairs, start_id, started) <= 1:
                pairs, complete = self._page_sentences(p, wait_next=True)  # nothing else to give: wait
            pg = self.page(p)
            if not started and pg.tail and pg.tail_joined and pg.tail["sentence"]["id"] == start_id:
                start_id = None  # that sentence now starts the next page
                p += 1
                continue
            for s, b in pairs:
                if not started and (start_id is None or s.id == start_id):
                    started = True
                if started:
                    items.append({"id": s.id, "page": p, "block": b.id if b else None,
                                  "kind": b.kind if b else "text", "bbox": b.bbox if b else None,
                                  "boxes": block_boxes(b) if b else [],
                                  "continues_from": b.continues_from if b else "",
                                  "prev_part": self._prev_part(b, s), "text": s.text, "spoken": s.spoken})
                    if len(items) >= count:
                        break
            if not started:  # start id not on this page (e.g. absorbed): start at its first sentence
                started, start_id = True, None
                continue
            if not complete:
                break  # the page-break sentence is decided when the next page is ready: the player asks again
            p += 1
        return items

    @staticmethod
    def _count_from(pairs, start_id, started):
        """How many of a page's sentences sequence() would take (from start_id, unless already started)."""
        if started or start_id is None:
            return len(pairs)
        ids = [s.id for s, _ in pairs]
        return len(ids) - ids.index(start_id) if start_id in ids else 0

    def _tail_target(self, pg, p):
        """The sentence that starts with page p's page-break fragment: the fragment itself until the next page is
        prepared, then the sentence it was joined into on that page."""
        if not pg.tail_joined:
            return pg.tail["sentence"]["id"]
        nxt = self.streamer.ready(p + 1)
        cont = next((b for b in nxt.blocks if b.continues_from == pg.tail["block"]), None) if nxt else None
        return cont.sentences[0].id if cont is not None and cont.sentences else pg.tail["sentence"]["id"]

    def _sentence_at(self, pg, p, block, text, at, before="", after=""):
        """The sentence of `block` under the clicked character. `text` is the clicked line of the page's text
        layer and `at` the clicked character's offset in it (None: the middle of the line); `before`/`after` the
        text-layer lines around it. A line usually holds the end of one sentence and the start of the next, so
        the line alone cannot tell which one was clicked: the line is found in the block's sentences joined
        together - with the lines around it when they are there too, as a short line ("M") occurs many times -
        and the click's position in it decides."""
        cands = [(x.id, letters(x.text)) for x in block.sentences]
        if pg.tail and pg.tail["block"] == block.id:  # the fragment moved to the next page is still printed here
            cands.append((self._tail_target(pg, p), letters(pg.tail["sentence"]["text"])))
        line = letters(text)
        if not line or not cands:
            return None
        if at is None:
            a = len(line) // 2
        else:  # the clicked character's place among the line's letters; a click on a "." or a space counts for
            # the letter before it (the sentence that "." ends), not for the next sentence's first letter
            a = len(letters(text[:at]))
            if at < len(text) and not letters(text[at]):
                a -= 1
        a = min(max(a, 0), len(line) - 1)
        stream = "".join(l for _, l in cands)
        pre, post = letters(before), letters(after)
        pos = None
        # the stretch around the click, found once; the lines before or after can belong to another block (a
        # paragraph's first or last line), so each side is also tried alone
        for find, lead in ((pre + line + post, len(pre)), (pre + line, len(pre)), (line + post, 0)):
            if len(find) > len(line):
                i = stream.find(find)
                if i >= 0 and stream.find(find, i + 1) < 0:
                    pos = i + lead + a
                    break
        if pos is None and (i := stream.find(line)) >= 0:  # the line alone
            pos = i + a
        if pos is None:  # the line differs from the index somewhere (symbols): the letters around the click
            lo = max(0, a - 12)
            probe = line[lo:a + 12]
            j = stream.find(probe) if len(probe) >= 8 else -1
            if j < 0:
                return None
            pos = j + (a - lo)
        end = 0
        for sid, lt in cands:
            end += len(lt)
            if pos < end:
                return sid
        return cands[-1][0]

    def locate(self, p, x, y, text="", at=None, before="", after=""):
        pg = self.page(p, priority=True)
        tail = pg.tail
        if tail and tail.get("bbox") and not any(b.id == tail["block"] for b in pg.blocks):
            # the page-break sentence was the only one in its block, which left the page with it: a click where
            # that block was is a click on that sentence (not on the nearest other block - DoRA p1: the ICML notice)
            if any(in_box(box, x, y, 3) for box in tail.get("boxes") or [tail["bbox"]]):
                if not pg.tail_joined:
                    return tail["sentence"]["id"]
                nxt = self.streamer.ready(p + 1)
                cont = next((x for x in nxt.blocks if x.continues_from == tail["block"]), None) if nxt else None
                if cont is not None and cont.sentences:
                    return cont.sentences[0].id
        block = locate(pg.blocks, x, y)
        if block is None:
            nxt = next((q for q in range(p + 1, self.n_pages + 1) if self.page(q).blocks), None)
            return self.page(nxt).blocks[0].sentences[0].id if nxt else None
        sid = self._sentence_at(pg, p, block, text, at, before, after)
        if sid is not None:
            return sid
        sentence = block.sentences[0]
        want = letters(text)
        if want and len(want) >= 4:
            for s in block.sentences:  # the sentence containing the clicked text
                if want in letters(s.text) or letters(s.text) in want:
                    return s.id
            # the first part of a sentence that goes on to the next page
            if pg.tail and pg.tail["block"] == block.id:
                head = letters(pg.tail["sentence"]["text"])
                if want in head or head in want:
                    if not pg.tail_joined:  # the next page is not prepared yet: sequence() waits for it and
                        return pg.tail["sentence"]["id"]  # goes on with the joined sentence if it is joined
                    nxt = self.streamer.ready(p + 1)
                    cont = next((b for b in nxt.blocks if b.continues_from == block.id), None) if nxt else None
                    if cont is not None and cont.sentences:
                        return cont.sentences[0].id
        return sentence.id


class Server:
    def __init__(self):
        self.sessions = {}
        self.engine = None
        self.narrator = None
        self.ready = threading.Event()
        self.voice_lock = threading.Lock()
        self.inflight, self.inflight_lock = {}, threading.Lock()  # sentences being synthesised right now
        self.audio_cache = OrderedDict()  # (doc, sid, speed) -> wav bytes, LRU
        self.prefetcher = ThreadPoolExecutor(max_workers=1)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        from pdf_reader.models import ensure
        from pdf_reader.read_aloud import Narrator
        ensure()  # first start of a fresh install: downloads the models (models.py)
        from pdf_reader.stream import Engine
        t = time.perf_counter()
        self.narrator = Narrator()
        self.engine = Engine()
        self.ready.set()
        print(f"models loaded in {time.perf_counter() - t:.1f}s; ready", flush=True)

    def open(self, data=None, path=None, name=None):
        self.ready.wait()
        if path:
            data = Path(path).read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        if doc_id not in self.sessions:
            folder = CACHE / doc_id
            folder.mkdir(parents=True, exist_ok=True)
            pdf_path = folder / "document.pdf"
            if not pdf_path.exists():
                pdf_path.write_bytes(data)
            name = Path(path).name if path else (name or "document.pdf")
            self.sessions[doc_id] = Session(doc_id, pdf_path, self.engine, name)
        self.activate(doc_id)
        s = self.sessions[doc_id]
        return {"doc": doc_id, "pages": s.n_pages, "name": s.name, "indexed": sorted(s.streamer.pages)}

    def activate(self, doc_id):
        """Only the document being read is indexed in the background; the others pause (CPU and memory are
        shared with the voice). They resume when they are used again."""
        if getattr(self, "active", None) == doc_id:
            return
        self.active = doc_id
        for other_id, s in self.sessions.items():
            s.streamer.pause(other_id != doc_id)

    def _synth(self, doc_id, sid, speed):
        """As _synth_now, but a sentence already being synthesised (the viewer asks for its audio and its timing
        at the same moment) is waited for instead of being made twice: the second request used to queue behind
        the first and its look-ahead, and the voice started only when both were done (2.9 s -> 6.8 s)."""
        key = (doc_id, sid, round(speed, 2))
        with self.inflight_lock:
            if key in self.audio_cache:
                self.audio_cache.move_to_end(key)
                return self.audio_cache[key]
            event = self.inflight.get(key)
            owner = event is None
            if owner:
                event = self.inflight[key] = threading.Event()
        if not owner:
            event.wait()
            got = self.audio_cache.get(key)
            return got if got is not None else self._synth_now(doc_id, sid, speed)
        try:
            return self._synth_now(doc_id, sid, speed)
        finally:
            with self.inflight_lock:
                del self.inflight[key]
            event.set()

    def _synth_now(self, doc_id, sid, speed):
        """(wav bytes, timing) for one sentence, cached. timing: {"duration", "words": [{"w", "start", "end",
        "printed"}], "printed": [printed words]} - when each spoken word is said and which printed word it is."""
        from pdf_reader.narration import PAUSES
        from pdf_reader.timing import spoken_to_printed, word_times, word_times_aligned
        key = (doc_id, sid, round(speed, 2))
        if key in self.audio_cache:
            self.audio_cache.move_to_end(key)
            return self.audio_cache[key]
        session = self.sessions[doc_id]
        item = session.item(sid)
        if not item:
            return None
        printed = item["text"]
        spoken, pace, pause = session.voice_unit(sid)
        with self.voice_lock:
            self.narrator.speed = speed
            chunks = self.narrator.chunks(spoken, pace)
            gap = self.narrator.silence(PAUSES[pause])  # the planned pause, at the end of this sentence's clip
            rate = self.narrator.rate
        speech = b"".join(a for _, a, _ in chunks)
        pcm = speech + gap
        tokens = spoken.split()
        times = None
        if all(al for _, _, al in chunks):  # exact: the voice's own phoneme durations
            times = word_times_aligned([(al, len(a) // 2) for _, a, al in chunks], tokens, rate)
        if times is None:
            times = word_times([(ph, len(a) / 2 / rate) for ph, a, _ in chunks], tokens)
        owners = spoken_to_printed(printed, tokens) if printed else [None] * len(tokens)
        timing = {"duration": len(pcm) / 2 / rate, "speech": len(speech) / 2 / rate, "printed": printed.split(),
                  "words": [{"w": t, "start": round(a, 3), "end": round(b, 3), "printed": o}
                            for t, (a, b), o in zip(tokens, times, owners)]}
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
        self.audio_cache[key] = (buf.getvalue(), timing)
        # keep the most recent clips up to AUDIO_CACHE_BYTES (a few minutes of speech): replay and the look-ahead
        # are served from here, older sentences are synthesised again if needed (~0.3 s each)
        while len(self.audio_cache) > 1 and sum(len(w) for w, _ in self.audio_cache.values()) > AUDIO_CACHE_BYTES:
            self.audio_cache.popitem(last=False)
        return self.audio_cache[key]

    def audio(self, doc_id, sid, speed, prefetch=True):
        # the listener is on this page: the indexer prepares it and the pages after it first, so the next
        # page is ready before playback gets there
        self.sessions[doc_id].streamer.focus(int(sid.split(".")[0]))
        got = self._synth(doc_id, sid, speed)
        if prefetch:  # synthesise what comes next while this sentence plays
            def ahead():
                for nxt in self.sessions[doc_id].sequence(sid, 3)[1:]:
                    self._synth(doc_id, nxt["id"], speed)
            self.prefetcher.submit(ahead)
        return got[0] if got else None

    def timing(self, doc_id, sid, speed):
        got = self._synth(doc_id, sid, speed)
        return got[1] if got else None


SERVER = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _allowed(self):
        """Local callers only (see the module docstring). Refuses with 403 and returns False otherwise."""
        port = self.server.server_address[1]
        host_ok = self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")
        origin = self.headers.get("Origin")
        origin_ok = (origin is None or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")
                     or origin.startswith(("chrome-extension://", "moz-extension://", "extension://")))
        if host_ok and origin_ok:
            LAST_REQUEST[0] = time.monotonic()
            return True
        self._send(403, {"error": "only local callers"})
        return False

    def do_OPTIONS(self):  # no CORS: a preflight from another site is refused
        self._send(403, {"error": "only local callers"})

    def do_POST(self):
        try:
            if not self._allowed():
                return
            url = urlparse(self.path)
            if url.path == "/api/shutdown":  # the extension's "Stop" button (local callers only, see _allowed)
                self._send(200, {"stopping": True})
                threading.Thread(target=lambda: (time.sleep(0.3), os._exit(0)), daemon=True).start()
                return
            if url.path != "/api/open":
                return self._send(404, {"error": "not found"})
            name = parse_qs(url.query).get("name", [None])[0]
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(200, SERVER.open(path=json.loads(body)["path"]))
            if not body.startswith(b"%PDF"):
                return self._send(400, {"error": "not a PDF"})
            return self._send(200, SERVER.open(data=body, name=name and Path(name).name))
        except Exception as e:
            self._send(500, {"error": repr(e)})

    def _static(self, base, rel):
        f = (base / rel).resolve()
        if base.resolve() not in f.parents or not f.is_file():  # no escaping the folder
            return self._send(404, {"error": "not found"})
        return self._send(200, f.read_bytes(), STATIC_TYPES.get(f.suffix, "application/octet-stream"))

    def do_GET(self):
        try:
            if not self._allowed():
                return
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = url.path.strip("/").split("/")
            if url.path in ("/", "/index.html"):
                return self._static(VIEWER, "index.html")
            if parts[0] == "viewer":
                return self._static(VIEWER, "/".join(parts[1:]))
            if parts[0] == "pdfjs":
                return self._static(PDFJS, "/".join(parts[1:]))
            if url.path == "/api/health":
                return self._send(200, {"ready": SERVER.ready.is_set(), "docs": list(SERVER.sessions),
                                        "code": CODE_VERSION})
            if len(parts) >= 4 and parts[:2] == ["api", "doc"]:
                s = SERVER.sessions.get(parts[2])
                if s is None:
                    return self._send(404, {"error": "unknown document; POST /api/open first"})
                SERVER.activate(s.id)
                kind = parts[3]
                if kind == "pdf":  # the file itself, for the viewer when it was opened by path
                    return self._send(200, (CACHE / s.id / "document.pdf").read_bytes(), "application/pdf")
                if kind == "page":
                    p = int(parts[4])
                    if not 1 <= p <= s.n_pages:
                        return self._send(404, {"error": f"page {p} of {s.n_pages}"})
                    pg = s.page(p, wait=q.get("wait", "1") != "0", priority=True)
                    if pg is None:
                        s.streamer.cursor = p  # not ready: make it next
                        return self._send(200, {"page": p, "ready": False})
                    return self._send(200, s.page_json(pg))
                if kind == "locate":
                    at = int(q["at"]) if q.get("at", "").isdigit() else None
                    sid = s.locate(int(q["page"]), float(q["x"]), float(q["y"]), q.get("text", ""), at,
                                   q.get("before", ""), q.get("after", ""))
                    return self._send(200, {"sentence": sid})
                if kind == "sequence":
                    return self._send(200, s.sequence(q["from"], int(q.get("count", 20))))
                if kind == "audio":
                    wav = SERVER.audio(s.id, parts[4], float(q.get("speed", 1.0)))
                    return self._send(200, wav, "audio/wav") if wav else self._send(404, {"error": "no sentence"})
                if kind == "timing":
                    t = SERVER.timing(s.id, parts[4], float(q.get("speed", 1.0)))
                    return self._send(200, t) if t else self._send(404, {"error": "no sentence"})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": repr(e)})


def main():
    global SERVER
    from pdf_reader.models import report_crashes
    report_crashes()
    ap = argparse.ArgumentParser(description="Local server for the PDF reader viewer / extension")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--idle-exit", type=float, default=20,
                    help="quit after this many minutes without any request (0: never). Open reader tabs keep it "
                         "alive; the browser extension starts it again when needed (papervoice-setup)")
    a = ap.parse_args()
    SERVER = Server()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"PDF reader on http://127.0.0.1:{a.port}  - open that address in Edge (models are loading...)",
          flush=True)
    if a.idle_exit > 0:
        def exit_when_idle():
            while True:
                time.sleep(30)
                if time.monotonic() - LAST_REQUEST[0] > a.idle_exit * 60:
                    print(f"no request for {a.idle_exit:g} minutes: stopping (frees its memory)", flush=True)
                    os._exit(0)
        threading.Thread(target=exit_when_idle, daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    main()
