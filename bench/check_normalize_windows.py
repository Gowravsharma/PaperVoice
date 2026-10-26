"""
check_normalize_windows.py - windowed normalisation (text_norm._normalize_sentence) against whole-sentence
normalisation, on every sentence of the indexed papers in out/cache. Reports identical outputs and the time saved.

Usage: uv run python bench/check_normalize_windows.py [out/cache]
"""
import json
import sys
import time
from pathlib import Path

from pdf_reader import text_norm as tn

tn._get_normalizer()
inputs = []
for page in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "out/cache").glob("*/page_*.json")):
    for b in json.loads(page.read_text(encoding="utf-8"))["blocks"]:
        if b["kind"] in ("text", "caption", "heading"):
            for s in b["sentences"]:
                x = tn.symbols(tn.prose(s["text"]))
                for part in __import__("re").split(r"(?<=[.!?;])\s+", x):
                    if tn.NEEDS_TN.search(part):
                        inputs.append(part)
inputs = list(dict.fromkeys(inputs))
t = time.perf_counter()
whole = []
for s in inputs:
    try:
        whole.append(tn._normalizer.normalize(s))
    except Exception:
        whole.append(None)
t_whole = time.perf_counter() - t
t = time.perf_counter()
windowed = []
for s in inputs:
    try:
        windowed.append(tn._normalize_sentence(s))
    except Exception:
        windowed.append(None)
t_win = time.perf_counter() - t
diff = [(s, a, b) for s, a, b in zip(inputs, whole, windowed) if a != b]
print(f"{len(inputs)} sentences to normalise: identical {len(inputs) - len(diff)}, different {len(diff)}; "
      f"time whole {t_whole:.1f} s, windowed {t_win:.1f} s ({t_whole / max(t_win, 1e-9):.1f}x faster)")
for s, a, b in diff[:12]:
    print(f"  input   : {s[:150]}\n  whole   : {a[:150] if a else a}\n  windowed: {b[:150] if b else b}\n")
