"""
compare_voices.py - Piper against Kokoro on the same text: time, memory, processor use, and blind listening samples.

Usage: uv run python bench/compare_voices.py out/audit/AFTER/<paper>_p5-5/05_piper_chunks.json [--seconds 60] [--runs 2]

Each engine runs in a process of its own (so one engine's memory cannot show up in the other's), on the same units
(the text papervoice sends to the voice, with its planned pauses: the file written by bench/audit_pipeline.py).
Per engine it reports the voice's load time, the resident memory after loading and at its peak, the time to
synthesise every unit (the path the server uses: Narrator.chunks), the real-time factor (compute seconds per second
of audio; below 1 = faster than speaking), the longest single unit, the share of the processor used, and the
delay before the first sound of the first unit. It also writes the first --seconds of the excerpt for each engine
as sample_A.wav / sample_B.wav in random order (the key is in KEY.json: open it only after listening).
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import wave
from pathlib import Path

FORMULA_PACE = 0.8  # formulas are read at 0.8 of the speed, as in the server (Session.voice_unit)


def child(engine, units_file, seconds, out_wav, runs):
    import psutil
    proc = psutil.Process()
    os.environ["PAPERVOICE_VOICE_ENGINE"] = engine
    units = json.loads(Path(units_file).read_text(encoding="utf-8"))
    base = proc.memory_info().rss
    t = time.perf_counter()
    from pdf_reader.narration import PAUSES
    from pdf_reader.read_aloud import make_narrator
    narrator = make_narrator()
    load = time.perf_counter() - t
    after_load = proc.memory_info().rss

    def pace(u):
        return FORMULA_PACE if u["kind"] == "formula" else 1.0

    narrator.chunks(units[0]["text_to_piper"], 1.0)  # warm-up (first call builds caches); not counted
    best = None
    for _ in range(runs):
        cpu0, wall0 = sum(proc.cpu_times()[:2]), time.perf_counter()
        per_unit, audio_s = [], 0.0
        pcm_all = []
        for u in units:
            t = time.perf_counter()
            chunks = narrator.chunks(u["text_to_piper"], pace(u))
            per_unit.append(time.perf_counter() - t)
            pcm = b"".join(c[1] for c in chunks)
            audio_s += len(pcm) / 2 / narrator.rate
            pcm_all.append(pcm + narrator.silence(PAUSES[u["pause_after"]]))
        wall = time.perf_counter() - wall0
        cpu = sum(proc.cpu_times()[:2]) - cpu0
        run = {"wall": wall, "audio_s": audio_s, "per_unit": per_unit, "cpu_cores": cpu / wall, "pcm": pcm_all}
        if best is None or wall < best["wall"]:
            best = run  # the fastest of the runs (the slower ones carry background noise)
    peak = proc.memory_info().peak_wset

    # the listening excerpt: units from the start until ~seconds of audio (pauses included)
    keep, total = [], 0.0
    for pcm in best["pcm"]:
        keep.append(pcm)
        total += len(pcm) / 2 / narrator.rate
        if total >= seconds:
            break
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(narrator.rate)
        w.writeframes(b"".join(keep))

    words = sum(len(u["text_to_piper"].split()) for u in units)
    print("RESULT " + json.dumps({
        "engine": engine, "sample_rate": narrator.rate, "units": len(units), "words": words,
        "load_s": round(load, 2), "ram_before_mb": round(base / 2**20), "ram_after_load_mb": round(after_load / 2**20),
        "ram_peak_mb": round(peak / 2**20), "synth_s": round(best["wall"], 1), "audio_s": round(best["audio_s"], 1),
        "rtf": round(best["wall"] / best["audio_s"], 3), "slowest_unit_s": round(max(best["per_unit"]), 2),
        "first_unit_s": round(best["per_unit"][0], 2), "cpu_cores_avg": round(best["cpu_cores"], 2),
        "logical_cpus": os.cpu_count(), "words_per_min": round(words / best["audio_s"] * 60)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--runs", type=int, default=2, help="timed passes over all units; the fastest is reported")
    ap.add_argument("--out", default="out/compare_voices")
    ap.add_argument("--child")
    ap.add_argument("--wav")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.units, a.seconds, a.wav, a.runs)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for engine in ("piper", "kokoro"):
        wav = out / f"{engine}.wav"
        print(f"running {engine} ...", flush=True)
        run = subprocess.run([sys.executable, __file__, a.units, "--child", engine, "--wav", str(wav),
                              "--seconds", str(a.seconds), "--runs", str(a.runs)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((l for l in run.stdout.splitlines() if l.startswith("RESULT ")), None)
        if run.returncode != 0 or not line:
            sys.exit(f"{engine} failed (exit {run.returncode}):\n{run.stderr[-1500:]}")
        results[engine] = json.loads(line[7:])
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = [("voice load (s)", "load_s"), ("RAM before load (MB)", "ram_before_mb"),
            ("RAM after load (MB)", "ram_after_load_mb"), ("RAM peak (MB)", "ram_peak_mb"),
            ("time to synthesise all (s)", "synth_s"), ("audio produced (s)", "audio_s"),
            ("real-time factor", "rtf"), ("slowest unit (s)", "slowest_unit_s"), ("first unit (s)", "first_unit_s"),
            ("processor cores used (avg)", "cpu_cores_avg"), ("speaking rate (words/min)", "words_per_min"),
            ("sample rate (Hz)", "sample_rate")]
    p, k = results["piper"], results["kokoro"]
    print(f"\n{p['units']} units, {p['words']} words; logical processors: {p['logical_cpus']}\n")
    print(f"{'':32}{'Piper':>10}{'Kokoro':>10}")
    for label, key in rows:
        print(f"{label:32}{p[key]:>10}{k[key]:>10}")

    order = ["piper", "kokoro"]
    random.shuffle(order)
    key = {}
    for letter, engine in zip("AB", order):
        (out / f"{engine}.wav").replace(out / f"sample_{letter}.wav")
        key[letter] = engine
    (out / "KEY.json").write_text(json.dumps(key), encoding="utf-8")
    print(f"\nlistening samples: {out / 'sample_A.wav'} and {out / 'sample_B.wav'} (key in KEY.json - open after listening)")


if __name__ == "__main__":
    main()
