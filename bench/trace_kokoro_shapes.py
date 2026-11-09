"""
trace_kokoro_shapes.py - Trace and print the EXACT tensor shapes at every stage
of Kokoro-82M for both the 106-character sentence and the 581-character formula.
"""
import onnx
import onnxruntime as rt
import numpy as np
from kokoro_onnx import Kokoro

def trace():
    m = onnx.load("models/kokoro/kokoro-v1.0.onnx", load_external_data=False)
    
    # Text 1: 106 characters
    text_106 = "The model reads research papers aloud, and the listener follows each sentence on the page as it is spoken."
    
    # Text 2: 581 characters (longest formula)
    import json
    units = json.load(open("out/audit/AFTER/a_probabilistic_numeric_method_p5-5/05_piper_chunks.json", encoding="utf-8"))
    text_581 = max(units, key=lambda u: len(u["text_to_piper"]))["text_to_piper"]
    
    k = Kokoro("models/kokoro/kokoro-v1.0.onnx", "models/kokoro/voices-v1.0.bin")
    
    # Trace with ORT profiling to get exact runtime shapes of every kernel
    for label, txt in [("106-Char Sentence", text_106), ("581-Char Formula", text_581)]:
        opts = rt.SessionOptions()
        opts.enable_profiling = True
        sess = rt.InferenceSession("models/kokoro/kokoro-v1.0.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
        k_trace = Kokoro.from_session(sess, "models/kokoro/voices-v1.0.bin")
        
        audio, sr = k_trace.create(txt, voice="af_heart", speed=1.0 if len(txt) < 200 else 0.8, lang="en-us")
        prof_file = sess.end_profiling()
        
        events = json.load(open(prof_file))
        node_events = {e["name"].split()[0]: e for e in events if e.get("cat") == "Node"}
        
        print(f"\n=======================================================")
        print(f"  EXACT TRACE: {label} ({len(txt)} chars)")
        print(f"=======================================================")
        print(f"Output Audio: {len(audio)} samples at {sr} Hz = {len(audio)/sr:.2f} seconds")
        
        # Key landmarks in the architecture
        landmarks = [
            ("Text Phonemes Input", "/encoder/bert/encoder/embedding/Cast_1_output_0"),
            ("Text Encoder Output (ALBERT)", "/encoder/bert/encoder/Slice_1"),
            ("Acoustic Features Input to Generator", "/decoder/decoder/generator/ups.0/ConvTranspose"),
            ("After Upsample 1 (ups.0: Stride 10)", "/decoder/decoder/generator/resblocks.0/convs1.0/Conv"),
            ("After Upsample 2 (ups.1: Stride 6)", "/decoder/decoder/generator/resblocks.3/convs1.0/Conv"),
            ("Signal Input to STFT", "/decoder/decoder/generator/STFT"),
            ("STFT Output", "/decoder/decoder/generator/Transpose_3"),
            ("Final Vocoder Output (istft: Stride 5)", "/decoder/decoder/generator/istft/stft/ConvTranspose"),
        ]
        
        for desc, name in landmarks:
            match = next((e for k, e in node_events.items() if name in k), None)
            if match:
                in_shape = match["args"].get("input_type_shape")
                out_shape = match["args"].get("output_type_shape")
                print(f"\n[Landmark] {desc}:")
                print(f"  Kernel: {match['name'].split()[0]}")
                print(f"  Input  Shape: {in_shape}")
                print(f"  Output Shape: {out_shape}")

if __name__ == "__main__":
    trace()
