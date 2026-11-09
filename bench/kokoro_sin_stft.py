import collections, json, math, sys
import onnx, onnxruntime as rt
from kokoro_onnx import Kokoro

m = onnx.load("models/kokoro/kokoro-v1.0.onnx", load_external_data=False)
sin = [n for n in m.graph.node if n.op_type == "Sin"]
print("Sin nodes:", len(sin), "| example names:", [n.name for n in sin[:3]])
pow_ = [n for n in m.graph.node if n.op_type == "Pow"]
print("Pow nodes:", len(pow_), "| example names:", [n.name for n in pow_[:2]])
stft = [n for n in m.graph.node if n.op_type == "STFT"]
print("STFT nodes:", len(stft), [n.name for n in stft])

opts = rt.SessionOptions(); opts.enable_profiling = True
sess = rt.InferenceSession("models/kokoro/kokoro-v1.0.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
k = Kokoro.from_session(sess, "models/kokoro/voices-v1.0.bin")
text = "The model reads research papers aloud, and the listener follows each sentence on the page as it is spoken."
a, sr = k.create(text, voice="af_heart", speed=1.0, lang="en-us")
path = sess.end_profiling()
ev = [e for e in json.load(open(path)) if e.get("cat") == "Node" and e.get("name", "").endswith("kernel_time")]
total = sum(e["dur"] for e in ev)
print(f"\naudio {len(a)/sr:.2f} s = {len(a)} samples at {sr} Hz; operator time {total/1e6:.2f} s")
for op in ("Sin", "STFT"):
    es = [e for e in ev if e["args"]["op_name"] == op]
    elems = 0
    for e in es:
        shapes = json.loads("[" + e["args"]["output_type_shape"].replace("'", '"') + "]") if False else e["args"]["output_type_shape"]
        # output_type_shape is a list of {type: [dims]}
        for s in shapes:
            for dims in s.values():
                elems += math.prod(dims)
    dur = sum(e["dur"] for e in es)
    print(f"{op:5}: {len(es)} nodes, {dur/1e6:.2f} s ({dur/total:.1%}), {elems/1e6:.1f} million output values"
          + (f" -> {dur/1e3/(elems/1e6):.1f} ns per value" if elems else ""))
big = sorted((e for e in ev if e["args"]["op_name"] == "Sin"), key=lambda e: -e["dur"])[:3]
for e in big: print("   slowest Sin:", e["dur"]/1e3, "ms, output", e["args"]["output_type_shape"])
s = [e for e in ev if e["args"]["op_name"] == "STFT"]
for e in s: print("   STFT:", e["dur"]/1e3, "ms, input", e["args"].get("input_type_shape"), "output", e["args"]["output_type_shape"])
