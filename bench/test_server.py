"""
test_server.py - end-to-end check of papervoice-server inside this process (on a spare port), then exit.

Usage: uv run python bench/test_server.py <pdf> [--port 8799]
Does what the extension and the viewer do: open the PDF (bytes + name), fetch it back, get a page, the reading
sequence, audio and word timings of the first sentence. Checks that foreign callers are refused (another
website's Origin, a DNS-rebinding Host), and reports the memory of the server and its Node helper.
"""
import argparse
import os
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import psutil

from pdf_reader import server


def call(base, path, data=None, headers=None, method=None):
    req = urllib.request.Request(base + path, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    t0 = time.perf_counter()
    server.SERVER = server.Server()
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{a.port}"
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    while json.loads(call(base, "/api/health")[2]).get("ready") is not True:
        time.sleep(0.5)
    print(f"server ready in {time.perf_counter() - t0:.1f} s")

    pdf = Path(a.pdf).read_bytes()
    st, _, body = call(base, "/api/open?name=" + Path(a.pdf).name, pdf, {"Content-Type": "application/pdf",
                                                                         "Origin": "chrome-extension://abc"})
    info = json.loads(body)
    check("extension opens a PDF", st == 200 and info.get("doc"), f"{info.get('pages')} pages, name {info.get('name')}")
    doc = info["doc"]
    st, ctype, body = call(base, f"/api/doc/{doc}/pdf")
    check("viewer fetches the PDF back", st == 200 and body == pdf and ctype == "application/pdf")
    st, _, body = call(base, "/?doc=x")
    check("viewer page served", st == 200 and b"viewer.js" in body)
    t = time.perf_counter()
    st, _, body = call(base, f"/api/doc/{doc}/page/1")
    page = json.loads(body)
    check("page 1 indexed", st == 200 and page.get("blocks"), f"{time.perf_counter() - t:.1f} s")
    first = page["blocks"][0]["sentences"][0]["id"]
    st, _, body = call(base, f"/api/doc/{doc}/sequence?from={first}&count=5")
    check("reading sequence", st == 200 and len(json.loads(body)) >= 1)
    t = time.perf_counter()
    st, ctype, wav = call(base, f"/api/doc/{doc}/audio/{first}?speed=1.0")
    check("audio of the first sentence", st == 200 and ctype == "audio/wav" and wav[:4] == b"RIFF",
          f"{len(wav) / 44100:.1f} s of audio in {time.perf_counter() - t:.1f} s")
    st, _, body = call(base, f"/api/doc/{doc}/timing/{first}?speed=1.0")
    timing = json.loads(body)
    check("word timings", st == 200 and timing.get("words"))
    check("timings are the voice's own (aligned voice loaded)", server.SERVER.narrator.aligned)
    ends = [w["end"] for w in timing["words"]]
    check("words in order, inside the speech", ends == sorted(ends) and ends[-1] <= timing["speech"] + 0.01,
          f"last word ends {ends[-1]:.2f} s, speech {timing['speech']:.2f} s, clip {timing['duration']:.2f} s")
    check("planned pause at the end of the clip", timing["duration"] - timing["speech"] >= 0.3)

    st, _, _ = call(base, "/api/health", headers={"Origin": "https://evil.example"})
    check("another website is refused (GET)", st == 403)
    st, _, _ = call(base, "/api/open", json.dumps({"path": a.pdf}).encode(),
                    {"Content-Type": "application/json", "Origin": "https://evil.example"})
    check("another website cannot open local files", st == 403)
    st, _, _ = call(base, "/api/health", headers={"Host": f"rebind.example:{a.port}"})
    check("DNS-rebinding Host is refused", st == 403)
    st, _, _ = call(base, "/api/open", method="OPTIONS", headers={"Origin": "https://evil.example"})
    check("no CORS preflight for other sites", st == 403)
    st, _, _ = call(base, "/api/open", b"not a pdf", {"Content-Type": "application/pdf"})
    check("non-PDF body rejected", st == 400)

    me = psutil.Process()
    kids = me.children(recursive=True)
    rss = me.memory_info().rss + sum(k.memory_info().rss for k in kids)
    print(f"memory: server {me.memory_info().rss / 2**20:.0f} MB + helpers "
          f"{sum(k.memory_info().rss for k in kids) / 2**20:.0f} MB = {rss / 2**20:.0f} MB; "
          f"torch loaded: {'torch' in sys.modules}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED", flush=True)
    # exit at once: the server's background workers (indexing, audio ahead of the reader) are still busy, and
    # shutting them down one by one only produces teardown noise
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
