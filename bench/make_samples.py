"""
make_samples.py - blind listening samples for tuning the voice: the same excerpt under different settings.

Usage: uv run python bench/make_samples.py out/audit/AFTER/<paper>_p5-5/05_piper_chunks.json [--seconds 45]

Reuses the units of an audit run (same text and planned pauses for every sample), so only the voice setting
differs. Files are named sample_A.wav, sample_B.wav, ... in random order; the key (which letter is which
setting) is written to KEY.json, to be opened only after listening. Also prints words per minute and the
synthesis time of each setting.
"""
import argparse
import json
import random
import time
import wave
from pathlib import Path

from piper import SynthesisConfig

from pdf_reader.narration import PAUSES
from pdf_reader.read_aloud import Narrator

SETTINGS = [  # one change at a time from the voice's defaults (noise_scale 0.667, noise_w 0.8)
    {"name": "speed 1.00 (length_scale 1.00)", "speed": 1.00},
    {"name": "speed 0.85 (length_scale 1.18)", "speed": 0.85},
    {"name": "speed 0.75 (length_scale 1.33)", "speed": 0.75},
    {"name": "speed 0.85, noise_w 0.6 (more even rhythm)", "speed": 0.85, "noise_w": 0.6},
    {"name": "speed 0.85, noise_w 1.0 (more varied rhythm)", "speed": 0.85, "noise_w": 1.0},
    {"name": "speed 0.85 for text, 0.72 for formulas (math read slower)", "speed": 0.85, "formula_speed": 0.72},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--start", type=int, default=0, help="index of the first unit of the excerpt")
    ap.add_argument("--out", default="out/samples")
    a = ap.parse_args()
    units = json.loads(Path(a.units).read_text(encoding="utf-8"))[a.start:]
    # the excerpt: units from the start until ~--seconds of audio at speed 1.0
    excerpt, total = [], 0.0
    for u in units:
        excerpt.append(u)
        total += sum(c["seconds"] for c in u["piper_chunks"])
        if total >= a.seconds:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "excerpt.txt").write_text("\n".join(f"[{u['kind']}, then {u['pause_after']} pause] {u['text_to_piper']}"
                                              for u in excerpt), encoding="utf-8")
    narrator = Narrator()
    letters = list("ABCDEFGH"[:len(SETTINGS)])
    random.seed(time.time_ns())
    random.shuffle(letters)
    key, words = {}, sum(len(u["text_to_piper"].split()) for u in excerpt)
    for letter, s in zip(letters, SETTINGS):
        narrator.speed = s["speed"]
        t = time.perf_counter()
        pcm, speech = [], 0.0
        for u in excerpt:
            speed = s.get("formula_speed", s["speed"]) if u["kind"] == "formula" else s["speed"]
            cfg = SynthesisConfig(length_scale=1 / speed, noise_w_scale=s.get("noise_w"), noise_scale=s.get("noise_scale"))
            audio = b"".join(c.audio_int16_bytes for c in narrator.voice.synthesize(u["text_to_piper"], syn_config=cfg))
            speech += len(audio) / 2 / narrator.rate
            pcm += [audio, narrator.silence(PAUSES[u["pause_after"]])]
        synth = time.perf_counter() - t
        with wave.open(str(out / f"sample_{letter}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(narrator.rate)
            w.writeframes(b"".join(pcm))
        key[letter] = {**s, "words_per_minute_while_speaking": round(words / speech * 60), "synthesis_s": round(synth, 1)}
    (out / "KEY.json").write_text(json.dumps(dict(sorted(key.items())), indent=1), encoding="utf-8")
    print(f"{len(excerpt)} units, {words} words; wrote {', '.join(f'sample_{l}.wav' for l in sorted(key))} "
          f"and KEY.json (open after listening) in {out}")
    for l in sorted(key):
        print(f"  sample_{l}: {key[l]['words_per_minute_while_speaking']} words/min, synthesis {key[l]['synthesis_s']} s")


if __name__ == "__main__":
    main()
