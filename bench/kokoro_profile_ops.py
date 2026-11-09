"""Time per ONNX operation type for one Kokoro model: python kprof.py <model file>."""
import collections, json, sys
import onnxruntime as rt
from kokoro_onnx import Kokoro

model = sys.argv[1]
text = "The model reads research papers aloud, and the listener follows each sentence on the page as it is spoken."
opts = rt.SessionOptions(); opts.enable_profiling = True; opts.profile_file_prefix = "kprof_" + model.split(".")[1]
sess = rt.InferenceSession(f"models/kokoro/{model}", sess_options=opts, providers=["CPUExecutionProvider"])
k = Kokoro.from_session(sess, "models/kokoro/voices-v1.0.bin")
k.create(text, voice="af_heart", speed=1.0, lang="en-us")              # warm-up (not separated; profile covers both)
k.create(text, voice="af_heart", speed=1.0, lang="en-us")
path = sess.end_profiling()
events = json.load(open(path))
by, count = collections.Counter(), collections.Counter()
for e in events:
    if e.get("cat") == "Node" and e.get("name", "").endswith("kernel_time"):
        op = e["args"]["op_name"]; by[op] += e["dur"]; count[op] += 1
total = sum(by.values())
print(f"{model}: {total/2/1e6:.2f} s of operator time per sentence (average of 2 runs); top operations:")
for op, d in by.most_common(9):
    print(f"   {op:28} {d/total:6.1%}   {count[op]//2:5d} nodes")
