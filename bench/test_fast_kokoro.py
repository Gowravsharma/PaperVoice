"""
test_fast_kokoro.py - Compare original kokoro-v1.0.onnx vs kokoro-v1.0.fast.onnx (with Conv1D instead of STFT).
Tests execution, speedup, and numerical audio similarity.
"""
import time
import numpy as np
import onnxruntime as rt
from kokoro_onnx import Kokoro

def compare():
    voices_path = "models/kokoro/voices-v1.0.bin"
    orig_model = "models/kokoro/kokoro-v1.0.onnx"
    fast_model = "models/kokoro/kokoro-v1.0.fast.onnx"
    
    text = "The model reads research papers aloud, and the listener follows each sentence on the page as it is spoken."
    
    print("Loading original model...")
    k_orig = Kokoro(orig_model, voices_path)
    print("Loading fast model...")
    k_fast = Kokoro(fast_model, voices_path)
    
    # Warm-up
    print("Warming up...")
    k_orig.create("Warm up", voice="af_heart", speed=1.0, lang="en-us")
    k_fast.create("Warm up", voice="af_heart", speed=1.0, lang="en-us")
    
    # Benchmark 5 runs each
    print("\nBenchmarking original model (with STFT)...")
    t_origs = []
    for _ in range(5):
        t0 = time.perf_counter()
        a_orig, sr = k_orig.create(text, voice="af_heart", speed=1.0, lang="en-us")
        t_origs.append(time.perf_counter() - t0)
    
    print("Benchmarking fast model (with Conv1D)...")
    t_fasts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a_fast, sr = k_fast.create(text, voice="af_heart", speed=1.0, lang="en-us")
        t_fasts.append(time.perf_counter() - t0)
        
    med_orig = np.median(t_origs)
    med_fast = np.median(t_fasts)
    
    audio_dur = len(a_orig) / sr
    print(f"\nText: {len(text)} chars -> {audio_dur:.2f} s of audio")
    print(f"Original compute time: {med_orig:.3f} s (RTF: {med_orig/audio_dur:.3f})")
    print(f"Fast compute time:     {med_fast:.3f} s (RTF: {med_fast/audio_dur:.3f})")
    print(f"Time saved per sentence: {(med_orig - med_fast)*1000:.1f} ms")
    print(f"Speedup: {med_orig / med_fast:.2f}x faster")
    
    # Compare waveforms
    min_len = min(len(a_orig), len(a_fast))
    a1, a2 = a_orig[:min_len], a_fast[:min_len]
    corr = np.corrcoef(a1, a2)[0, 1]
    mae = np.mean(np.abs(a1 - a2))
    print(f"\nWaveform Pearson correlation: {corr:.6f} (1.000000 = identical sound)")
    print(f"Mean absolute difference:     {mae:.6f}")
    if corr > 0.999:
        print("PASS: Audio output is identical to human ear!")

if __name__ == "__main__":
    compare()
