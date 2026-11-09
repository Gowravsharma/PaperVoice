import os
import wave
from pathlib import Path
from pdf_reader.read_aloud import make_narrator

out_dir = Path("out")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "sample_fast_conv1d.wav"

narrator = make_narrator()
print(f"Narrator Engine: {type(narrator).__name__}")
print(f"Voice Name:      {getattr(narrator, 'voice_name', getattr(narrator, 'name', 'default'))}")
print(f"Sample Rate:     {narrator.rate} Hz")

paragraph = (
    "Recent advances in neural text to speech have demonstrated that non-autoregressive models "
    "can achieve high audio fidelity while maintaining low latency on personal computers. "
    "In this paper, we evaluate the acoustic properties of neural vocoders, replacing the "
    "slow Short-Time Fourier Transform with an exact one-dimensional convolution. "
    "This eliminates scalar compute bottlenecks while preserving pristine speech clarity."
)

print(f"\nSynthesizing paragraph ({len(paragraph.split())} words)...")
chunks = narrator.chunks(paragraph, pace=1.0)
speech = b"".join(a for _, a, _ in chunks)

with wave.open(str(out_file), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(narrator.rate)
    w.writeframes(speech)

duration = len(speech) / 2 / narrator.rate
print(f"SUCCESS: Saved sample audio to {out_file.resolve()} ({duration:.2f} s, {len(speech)} bytes)")

