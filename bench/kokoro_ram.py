"""Where Kokoro's memory goes: python kram.py <model> <arena 1|0> -> one RESULT line (MB, resident set)."""
import json, sys, time
import onnxruntime as rt, psutil
from kokoro_onnx import Kokoro

model, arena = sys.argv[1], sys.argv[2] == "1"
units = json.load(open("out/audit/AFTER/a_probabilistic_numeric_method_p5-5/05_piper_chunks.json", encoding="utf-8"))
short = min(units, key=lambda u: len(u["text_to_piper"]))
long_ = max(units, key=lambda u: len(u["text_to_piper"]))
p = psutil.Process()
mb = lambda: round(p.memory_info().rss / 2**20)
base = mb()
opts = rt.SessionOptions()
opts.enable_cpu_mem_arena = arena
sess = rt.InferenceSession(f"models/kokoro/{model}", sess_options=opts, providers=["CPUExecutionProvider"])
k = Kokoro.from_session(sess, "models/kokoro/voices-v1.0.bin")
loaded = mb()
r = {"model": model, "arena": arena, "python+libs": base, "after_load": loaded}
for name, u in (("short", short), ("long", long_)):
    t = time.perf_counter()
    a, sr = k.create(u["text_to_piper"], voice="af_heart", speed=1.0, lang="en-us")
    r[name] = {"chars": len(u["text_to_piper"]), "audio_s": round(len(a) / sr, 1), "took_s": round(time.perf_counter() - t, 1),
               "rss_after": mb(), "peak_so_far": round(p.memory_info().peak_wset / 2**20)}
print("RESULT " + json.dumps(r))
