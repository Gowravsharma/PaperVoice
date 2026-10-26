"""
ort_lean.py - ONNX Runtime sessions that give memory back between runs (imported first by the pipeline).

By default ONNX Runtime keeps a memory arena per session: buffers used for one page stay reserved for the next,
so the process grew by ~90 MB per converted page (bench/memory_breakdown.py). With the arena and the memory
pattern planner off, buffers are allocated per run and freed after it; on this CPU-only pipeline the speed cost
is small next to the models' compute. Applied to every session created after import - Docling's layout engine,
RapidOCR and the Piper voice create theirs through onnxruntime.InferenceSession too.
"""
import onnxruntime as ort

_Base = ort.InferenceSession


class LeanInferenceSession(_Base):
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        so = sess_options or ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        super().__init__(path_or_bytes, sess_options=so, providers=providers, provider_options=provider_options,
                         **kwargs)


if ort.InferenceSession is _Base:
    ort.InferenceSession = LeanInferenceSession
