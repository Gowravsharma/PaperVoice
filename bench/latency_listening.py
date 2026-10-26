"""
latency_listening.py - the waits a listener hears, measured in real time (in this process, no browser).

Usage: uv run python bench/latency_listening.py <pdf> [--sentences 40] [--fresh] [--speed 0.8]

Plays the reading order the way the viewer does: asks for a sentence's audio, "plays" it (waits for its length),
then asks for the next. Each request's wait is what the listener would hear as silence beyond the planned pause:
  - time to first sound after opening the document (and clicking the first sentence)
  - every gap > 0.1 s between sentences, with where it happened (and whether a page was being indexed)
--fresh: empty cache, so pages are indexed while listening (a newly opened paper).
"""
import argparse
import io
import os
import statistics
import time
import wave
from pathlib import Path

from pdf_reader import server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--sentences", type=int, default=40)
    ap.add_argument("--speed", type=float, default=0.8)
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    if a.fresh:
        import tempfile
        server.CACHE = Path(tempfile.mkdtemp())
    t0 = time.perf_counter()
    server.SERVER = srv = server.Server()
    srv.ready.wait()
    t_ready = time.perf_counter() - t0
    t = time.perf_counter()
    info = srv.open(data=Path(a.pdf).read_bytes(), name=Path(a.pdf).name)
    s = srv.sessions[info["doc"]]
    first = s.page(1, priority=True).blocks[0].sentences[0].id
    seq = s.sequence(first, a.sentences)
    waits, where = [], []
    for i, it in enumerate(seq):
        asked = time.perf_counter()
        wav = srv.audio(s.id, it["id"], a.speed)
        srv.timing(s.id, it["id"], a.speed)
        wait = time.perf_counter() - asked
        if i == 0:
            first_sound = time.perf_counter() - t
            print(f"engine ready {t_ready:.1f} s; first sound {first_sound:.1f} s after opening "
                  f"(page 1 indexed + first sentence)", flush=True)
        else:
            waits.append(wait)
            if wait > 0.1:
                where.append(f"  gap {wait:4.2f} s before sentence {i + 1} (page {it['page']}, "
                             f"{len(s.streamer.pages)} pages indexed): {it['text'][:60]!r}")
        with wave.open(io.BytesIO(wav)) as w:
            time.sleep(w.getnframes() / w.getframerate())  # the listener hears it
    print(f"{len(waits)} transitions: {sum(w > 0.1 for w in waits)} audible gaps (> 0.1 s), "
          f"total {sum(w for w in waits if w > 0.1):.1f} s, longest {max(waits):.2f} s, "
          f"median wait {statistics.median(waits) * 1000:.0f} ms", flush=True)
    print("\n".join(where[:15]))
    os._exit(0)


if __name__ == "__main__":
    main()
