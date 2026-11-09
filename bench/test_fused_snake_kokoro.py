"""
test_fused_snake_kokoro.py - End-to-end benchmark of Kokoro with Fused Snake Custom Operator.
Compares:
  1. kokoro-v1.0.fast.onnx (Conv1D STFT, unfused Snake)
  2. kokoro-v1.0.fused_snake.onnx (Conv1D STFT, 48 fused Snake activations via tools/snake_op.dll)
"""
import time
import pathlib
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

def create_fused_session(model_path="models/kokoro/kokoro-v1.0.fused_snake.onnx", dll_path="tools/snake_op.dll"):
    so = ort.SessionOptions()
    so.register_custom_ops_library(str(pathlib.Path(dll_path).resolve()))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
    return session

def main():
    voices_path = "models/kokoro/voices-v1.0.bin"
    test_text = "Optimization of neural vocoder inference using fused periodic activations on CPU."

    print("Loading baseline model (kokoro-v1.0.fast.onnx)...")
    k_fast = Kokoro("models/kokoro/kokoro-v1.0.fast.onnx", voices_path)

    print("Loading fused Snake model (kokoro-v1.0.fused_snake.onnx) with snake_op.dll...")
    fused_sess = create_fused_session()
    k_fused = Kokoro.from_session(fused_sess, voices_path)

    # Warm-up runs
    print("Warming up...")
    s_fast, _ = k_fast.create(test_text, voice="af_heart", speed=1.0)
    s_fused, _ = k_fused.create(test_text, voice="af_heart", speed=1.0)

    # 1. Audio fidelity comparison
    min_len = min(len(s_fast), len(s_fused))
    corr = np.corrcoef(s_fast[:min_len], s_fused[:min_len])[0, 1]
    max_diff = np.max(np.abs(s_fast[:min_len] - s_fused[:min_len]))
    mean_diff = np.mean(np.abs(s_fast[:min_len] - s_fused[:min_len]))

    print("\n--- Audio Fidelity (End-to-End Speech Waveform) ---")
    print(f"Sample Length:            {min_len} samples ({min_len / 24000:.2f} s audio)")
    print(f"Pearson Correlation (r):  {corr:.6f}")
    print(f"Mean Absolute Diff:       {mean_diff:.6e}")
    print(f"Max Absolute Diff:        {max_diff:.6e}")

    # 2. Timing benchmark (5 runs)
    N = 5
    print(f"\nBenchmarking over {N} runs...")
    t0 = time.perf_counter()
    for _ in range(N):
        k_fast.create(test_text, voice="af_heart", speed=1.0)
    t_fast = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        k_fused.create(test_text, voice="af_heart", speed=1.0)
    t_fused = (time.perf_counter() - t0) / N

    print("\n--- End-to-End Latency Results ---")
    print(f"Fast Kokoro (Conv1D STFT, unfused Snake):    {t_fast * 1000.0:.1f} ms  (RTF: {t_fast / (min_len / 24000):.3f})")
    print(f"Fused Snake Kokoro (Conv1D STFT + Fused):    {t_fused * 1000.0:.1f} ms  (RTF: {t_fused / (min_len / 24000):.3f})")
    print(f"Speedup:                                     {t_fast / t_fused:.2f}x faster!")
    print(f"Time saved per sentence:                     {(t_fast - t_fused) * 1000.0:.1f} ms")

if __name__ == '__main__':
    main()
