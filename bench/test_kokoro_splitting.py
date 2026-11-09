"""
test_kokoro_splitting.py - Measure chunk splitting on the longest formula unit:
time to first sound, piece-by-piece synthesis times, peak RAM, and total time.
"""
import json
import re
import sys
import time
import psutil
import numpy as np

def split_clauses(text, max_chars=180):
    """Split text into natural clauses of at most max_chars.
    Prefers punctuation (comma, semicolon, colon), then conjunctions, then word boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Pattern to find candidate split points:
    # 1. punctuation followed by space
    # 2. math/prose connective words set off by spaces
    # 3. plain spaces
    words = text.split(" ")
    cur = []
    cur_len = 0

    for w in words:
        w_len = len(w) + (1 if cur else 0)
        # If adding w exceeds max_chars and we already have words:
        if cur_len + w_len > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += w_len

    if cur:
        chunks.append(" ".join(cur))
    return chunks

def main():
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("kokoro-onnx not installed")
        return

    units = json.load(open("out/audit/AFTER/a_probabilistic_numeric_method_p5-5/05_piper_chunks.json", encoding="utf-8"))
    long_unit = max(units, key=lambda u: len(u["text_to_piper"]))
    text = long_unit["text_to_piper"]
    print(f"Longest unit: {len(text)} characters\n")

    k = Kokoro("models/kokoro/kokoro-v1.0.onnx", "models/kokoro/voices-v1.0.bin")
    
    # Warm-up
    k.create("Hello world", voice="af_heart", speed=1.0, lang="en-us")

    # 1. Monolithic run (current behavior)
    proc = psutil.Process()
    ram0 = proc.memory_info().peak_wset / 2**20
    t0 = time.perf_counter()
    samples_mono, sr = k.create(text, voice="af_heart", speed=0.8, lang="en-us")
    time_mono = time.perf_counter() - t0
    audio_s_mono = len(samples_mono) / sr
    ram_mono = proc.memory_info().peak_wset / 2**20

    print("=== 1. Monolithic (Current) ===")
    print(f"Time to first sound: {time_mono:.2f} s")
    print(f"Total audio produced: {audio_s_mono:.2f} s")
    print(f"Synthesis time: {time_mono:.2f} s (RTF: {time_mono / audio_s_mono:.3f})")
    print(f"Peak RAM: {ram_mono:.1f} MB\n")

    for target_chars in (100, 130, 160):
        chunks = split_clauses(text, max_chars=target_chars)
        print(f"\n=== Testing max_chars = {target_chars} ({len(chunks)} pieces) ===")
        t0 = time.perf_counter()
        first_sound = None
        continuous = True
        accum_audio = 0.0
        accum_compute = 0.0
        for i, c in enumerate(chunks, 1):
            tp0 = time.perf_counter()
            s_piece, sr = k.create(c, voice="af_heart", speed=0.8, lang="en-us")
            took = time.perf_counter() - tp0
            dur = len(s_piece) / sr
            if i == 1:
                first_sound = took
            else:
                # Did compute take longer than the audio of previous pieces?
                if accum_compute + took > accum_audio:
                    continuous = False
            accum_compute += took
            accum_audio += dur
            print(f"  Piece {i} ({len(c):3d} chars): synth {took:5.2f} s -> audio {dur:5.2f} s")
        print(f"  >> TTFS: {first_sound:.2f} s | Total audio: {accum_audio:.1f} s | Continuous buffer: {continuous}")


if __name__ == "__main__":
    main()
