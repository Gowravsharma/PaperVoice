"""One Kokoro configuration: python kspeed.py <model file> <threads or 0 for the default> -> one RESULT line."""
import json, os, sys, time
import numpy as np, onnxruntime as rt, psutil
from kokoro_onnx import Kokoro

model, threads = sys.argv[1], int(sys.argv[2])
units = json.load(open("out/audit/AFTER/a_probabilistic_numeric_method_p5-5/05_piper_chunks.json", encoding="utf-8"))
proc = psutil.Process()
opts = rt.SessionOptions()
if threads:
    opts.intra_op_num_threads = threads
t = time.perf_counter()
sess = rt.InferenceSession(f"models/kokoro/{model}", sess_options=opts, providers=["CPUExecutionProvider"])
k = Kokoro.from_session(sess, "models/kokoro/voices-v1.0.bin")
load = time.perf_counter() - t
outs = [o.name for o in sess.get_outputs()]
k.create(units[0]["text_to_piper"], voice="af_heart", speed=1.0, lang="en-us")  # warm-up
wall0, cpu0 = time.perf_counter(), sum(proc.cpu_times()[:2])
audio, per = 0.0, []
for u in units:
    t = time.perf_counter()
    a, sr = k.create(u["text_to_piper"], voice="af_heart", speed=0.8 if u["kind"] == "formula" else 1.0, lang="en-us")
    per.append(time.perf_counter() - t); audio += len(a) / sr
wall = time.perf_counter() - wall0
print("RESULT " + json.dumps({"model": model, "threads": threads or "default", "outputs": outs, "load_s": round(load, 1),
      "synth_s": round(wall, 1), "audio_s": round(audio, 1), "rtf": round(wall / audio, 3),
      "slowest_s": round(max(per), 1), "cores": round((sum(proc.cpu_times()[:2]) - cpu0) / wall, 2),
      "ram_peak_mb": round(proc.memory_info().peak_wset / 2**20)}))
