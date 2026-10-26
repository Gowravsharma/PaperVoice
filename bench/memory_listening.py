"""
memory_listening.py - memory of the server during a listening session (in this process, no browser).

Usage: uv run python bench/memory_listening.py <pdf> [--sentences 80]
Opens the PDF as the extension does, then asks for each sentence's audio and timing in reading order with
three sentences ahead, as the viewer does, while the background indexer works through the document. A sampler
thread records resident memory every 0.2 s; the peak and what was happening at the time are reported.
"""
import argparse
import threading
import time
from pathlib import Path

import psutil

from pdf_reader import server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--sentences", type=int, default=80)
    ap.add_argument("--fresh", action="store_true", help="index from scratch (empty cache), as for a new PDF")
    a = ap.parse_args()
    if a.fresh:
        import tempfile
        server.CACHE = Path(tempfile.mkdtemp())
    me = psutil.Process()
    phase, samples, stop = ["start"], [], threading.Event()

    def sampler():
        while not stop.is_set():
            m = me.memory_info()
            samples.append((time.perf_counter(), m.rss / 2**20, getattr(m, "private", m.rss) / 2**20, phase[0]))
            time.sleep(0.2)
    threading.Thread(target=sampler, daemon=True).start()

    phase[0] = "loading"
    server.SERVER = srv = server.Server()
    srv.ready.wait()
    phase[0] = "open"
    info = srv.open(data=Path(a.pdf).read_bytes(), name=Path(a.pdf).name)
    s = srv.sessions[info["doc"]]
    first = s.page(1).blocks[0].sentences[0].id
    seq = s.sequence(first, a.sentences)
    t0 = time.perf_counter()
    for i, it in enumerate(seq):
        phase[0] = f"sentence {i + 1} (page {it['page']})"
        srv.audio(s.id, it["id"], 0.9)  # what the viewer asks for, with the server's own look-ahead
        srv.timing(s.id, it["id"], 0.9)
        time.sleep(0.5)  # a listener needs seconds per sentence; this keeps the replay short but not instant
        if i % 10 == 9:
            r = me.memory_info().rss / 2**20
            kids = sum(k.memory_info().rss for k in me.children(recursive=True)) / 2**20
            print(f"{i + 1:3d} sentences, page {it['page']:2d}, pages indexed {len(s.streamer.pages):2d}: "
                  f"server {r:5.0f} MB + engine process and Node {kids:4.0f} MB", flush=True)
    stop.set()
    peak = max(samples, key=lambda x: x[1])
    print(f"listened {len(seq)} sentences in {time.perf_counter() - t0:.0f} s; "
          f"peak rss {peak[1]:.0f} MB (private {peak[2]:.0f} MB) during: {peak[3]}; "
          f"end rss {samples[-1][1]:.0f} MB; audio clips cached: {len(srv.audio_cache)}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
