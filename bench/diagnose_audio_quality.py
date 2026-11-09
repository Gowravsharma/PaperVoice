import time
import numpy as np
import wave
import onnxruntime as ort
from kokoro_onnx import Kokoro
import pathlib

def write_wav(path, samples, rate=24000):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)

text = "This is a test of speech quality to verify clarity and high frequency fidelity."
voices_path = "models/kokoro/voices-v1.0.bin"

out_dir = pathlib.Path("out/audio_quality_test")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Original Baseline Kokoro
print("1. Running Original Baseline kokoro-v1.0.onnx...")
k_orig = Kokoro("models/kokoro/kokoro-v1.0.onnx", voices_path)
s_orig, sr = k_orig.create(text, voice="af_heart", speed=1.0)
write_wav(out_dir / "1_orig_baseline.wav", s_orig, sr)

# 2. Fast Kokoro (Conv1D STFT)
print("2. Running kokoro-v1.0.fast.onnx (Conv1D STFT)...")
k_fast = Kokoro("models/kokoro/kokoro-v1.0.fast.onnx", voices_path)
s_fast, _ = k_fast.create(text, voice="af_heart", speed=1.0)
write_wav(out_dir / "2_conv1d_fast.wav", s_fast, sr)

# 3. Fused Snake Kokoro (with current snake_op.dll)
print("3. Running kokoro-v1.0.fused_snake.onnx...")
so = ort.SessionOptions()
so.register_custom_ops_library(str(pathlib.Path("tools/snake_op.dll").resolve()))
sess_fused = ort.InferenceSession("models/kokoro/kokoro-v1.0.fused_snake.onnx", so, providers=['CPUExecutionProvider'])
k_fused = Kokoro.from_session(sess_fused, voices_path)
s_fused, _ = k_fused.create(text, voice="af_heart", speed=1.0)
write_wav(out_dir / "3_fused_snake.wav", s_fused, sr)

# Compare waveforms
def compare(name, a, ref):
    min_len = min(len(a), len(ref))
    a1, a2 = a[:min_len], ref[:min_len]
    corr = np.corrcoef(a1, a2)[0, 1]
    mae = np.mean(np.abs(a1 - a2))
    # SNR in dB
    noise = a1 - a2
    signal_power = np.mean(a2 ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"{name:30} | Corr: {corr:.6f} | MAE: {mae:.6f} | SNR: {snr:5.1f} dB | Len: {len(a)} vs {len(ref)}")

print("\n--- Audio Comparison against 1_orig_baseline ---")
compare("1_orig_baseline", s_orig, s_orig)
compare("2_conv1d_fast", s_fast, s_orig)
compare("3_fused_snake", s_fused, s_orig)
compare("3_fused_snake vs 2_fast", s_fused, s_fast)
