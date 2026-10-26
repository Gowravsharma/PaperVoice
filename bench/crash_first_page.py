"""
crash_first_page.py - how often the first page conversion of a fresh process crashes (one run per process).

Usage: uv run python bench/crash_first_page.py <pdf> --runs 10 [--voice]
Each run starts a new Python process that loads the reading engine (and the voice with --voice, as the server
does), converts page 1 and exits. Prints how many runs crashed (exit code other than 0; Windows reports heap
corruption as 0xc0000374 / exit 127 under uv).
"""
import argparse
import subprocess
import sys
import time

CHILD = r"""
import sys
from pdf_reader.pdfium_safe import open_document
from pdf_reader.script import new_stats
if "--voice" in sys.argv:
    from pdf_reader.read_aloud import Narrator
    Narrator()
from pdf_reader.stream import Engine
eng = Engine(4)
pdf = open_document(sys.argv[1])
eng.index_page(sys.argv[1], pdf, 1, False, {"keep_references": False, "in_refs": False}, new_stats())
print("ok", flush=True)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--voice", action="store_true")
    a = ap.parse_args()
    crashed = 0
    for i in range(a.runs):
        t = time.perf_counter()
        r = subprocess.run([sys.executable, "-X", "faulthandler", "-c", CHILD, a.pdf] + (["--voice"] if a.voice else []),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        bad = r.returncode != 0 or "ok" not in r.stdout
        crashed += bad
        note = next((l for l in r.stderr.splitlines() if "exception: code" in l or "Error" in l), "")
        print(f"run {i + 1:2d}: {'CRASH' if bad else 'ok   '} ({time.perf_counter() - t:4.1f} s) {note}", flush=True)
    print(f"{crashed}/{a.runs} runs crashed ({'with' if a.voice else 'without'} the voice loaded)")


if __name__ == "__main__":
    main()
