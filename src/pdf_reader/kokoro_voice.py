"""
kokoro_voice.py - an optional second voice, Kokoro-82M, that can be plugged in instead of Piper (for testing).

    uv pip install kokoro-onnx                       # once; not part of the normal install
    set PAPERVOICE_VOICE_ENGINE=kokoro               # then start papervoice-server / papervoice-read as usual
    set PAPERVOICE_KOKORO_MODEL=kokoro-v1.0.extreme.onnx  # default: fast Conv1D + AVX2 Snake + Fused AdaIN
    set PAPERVOICE_KOKORO_MODEL=kokoro-v1.0.fast.onnx     # optional: fast Conv1D model (portable, no custom ops)
    set PAPERVOICE_KOKORO_MODEL=kokoro-v1.0.onnx          # optional: unmodified baseline
    set PAPERVOICE_KOKORO_MODEL=kokoro-v1.0.fused_snake.onnx  # optional: AVX2 fused Snake model (research)
    set PAPERVOICE_KOKORO_MODEL=kokoro-v1.0.int8.onnx     # optional: the smaller (92 MB) quantised model

Without the variable nothing changes: read_aloud.make_narrator() builds the Piper voice as before.

KokoroNarrator offers what the server and papervoice-read use of read_aloud.Narrator: speed, rate, silence(),
sentences() and chunks(). Differences from Piper: the audio is 24 kHz (Piper's voice: 22.05 kHz; everything reads
`rate`), and Kokoro does not output its phoneme durations, so the word highlight uses the estimated timing
(timing.word_times) instead of the exact one. The two model files (~350 MB) are downloaded into models/kokoro/ on
first use, from the kokoro-onnx project's release.
"""
import os
import pathlib
import re
import threading

import numpy as np

from pdf_reader.models import ROOT, MODELS_DIR, _download, sha256

KOKORO_DIR = MODELS_DIR / "kokoro"
SNAKE_DLL = ROOT / "tools" / "snake_op.dll"
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
VOICES_FILE = "voices-v1.0.bin"
# SHA-256 of verified Kokoro models and extensions
SHA256 = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "kokoro-v1.0.fast.onnx": "b330e0a1dbd130b26bddf3fb3585ea9168a3679b1b29ef13c8412b6d4eed031d",
    "kokoro-v1.0.extreme.onnx": "186242c7c87edd3b7e22ee240685e198c2bfc564b4b57fae6cc838806c5afa5b",
    "kokoro-v1.0.fused_snake.onnx": "1abf477b767d56639f154c7a66ff3b55baa32454e043e688137403ad027b94f1",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    "snake_op.dll": "568e4c9fb3419703a06dc5c71cb4011d6eba4b1d1571c5bb9747ded1d6327660",
}


def _patch_kokoro_stft(src_path, dst_path):
    """Replace the slow STFT node in kokoro-v1.0.onnx with a fast 1D convolution."""
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src_path))
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    consts = {
        n.output[0]: numpy_helper.to_array(n.attribute[0].t)
        for n in m.graph.node
        if n.op_type == "Constant" and len(n.attribute) and n.attribute[0].name == "value"
    }
    get_val = lambda name: inits.get(name, consts.get(name))

    win = get_val('/decoder/decoder/generator/Constant_17_output_0')
    win_size = len(win)  # 20
    hop = int(get_val('/decoder/decoder/generator/Constant_15_output_0'))  # 5
    n_bins = win_size // 2 + 1  # 11

    # Precompute DFT basis filter weights: shape [22, 1, 20]
    k = np.arange(n_bins)[:, None]
    n = np.arange(win_size)[None, :]
    angles = 2 * np.pi * k * n / win_size
    cos_w = (np.cos(angles) * win).astype(np.float32)
    sin_w = (-np.sin(angles) * win).astype(np.float32)
    conv_w = np.concatenate([cos_w, sin_w], axis=0)[:, None, :]

    weight_name = "/decoder/decoder/generator/stft_conv_weight"
    w_tensor = numpy_helper.from_array(conv_w, name=weight_name)
    m.graph.initializer.append(w_tensor)

    axes_name = "/decoder/decoder/generator/stft_unsqueeze_axes"
    axes_tensor = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
    m.graph.initializer.append(axes_tensor)

    sig_input = '/decoder/decoder/generator/Reshape_3_output_0'
    unsqueezed_sig = '/decoder/decoder/generator/stft_sig_unsqueezed'
    unsqueeze_node = helper.make_node(
        'Unsqueeze',
        inputs=[sig_input, axes_name],
        outputs=[unsqueezed_sig],
        name='/decoder/decoder/generator/stft_unsqueeze',
    )

    conv_output = '/decoder/decoder/generator/stft_conv_output'
    conv_node = helper.make_node(
        'Conv',
        inputs=[unsqueezed_sig, weight_name],
        outputs=[conv_output],
        strides=[hop],
        pads=[0, 0],
        name='/decoder/decoder/generator/stft_conv',
    )

    real_out = '/decoder/decoder/generator/Gather_4_output_0'
    imag_out = '/decoder/decoder/generator/Gather_5_output_0'
    split_node = helper.make_node(
        'Split',
        inputs=[conv_output],
        outputs=[real_out, imag_out],
        axis=1,
        num_outputs=2,
        name='/decoder/decoder/generator/stft_split',
    )

    nodes_to_remove = {
        '/decoder/decoder/generator/STFT',
        '/decoder/decoder/generator/Transpose_3',
        '/decoder/decoder/generator/Gather_4',
        '/decoder/decoder/generator/Gather_5',
    }
    unused_inits = {
        '/decoder/decoder/generator/Constant_15_output_0',
        '/decoder/decoder/generator/Constant_16_output_0',
        '/decoder/decoder/generator/Constant_17_output_0',
    }
    clean_inits = [t for t in m.graph.initializer if t.name not in unused_inits]
    del m.graph.initializer[:]
    m.graph.initializer.extend(clean_inits)

    stft_idx = next(i for i, node in enumerate(m.graph.node) if node.name == '/decoder/decoder/generator/STFT')
    new_nodes = [node for node in m.graph.node if node.name not in nodes_to_remove]
    new_nodes[stft_idx:stft_idx] = [unsqueeze_node, conv_node, split_node]
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)

    # Fix DC imaginary branch cut in atan2 decomposition:
    # Conv1D yields exact 0.0 for DC imag, which evaluated to 0.0 > 0.0 -> False,
    # causing a -pi phase flip across all speech frames (metallic distortion).
    # Shifting Constant_20 to -eps (-1.1920929e-07) forces exact 0.0 into the +pi branch.
    c20_name = '/decoder/decoder/generator/Constant_20_output_0'
    c20_patched = False
    for t in m.graph.initializer:
        if t.name == c20_name:
            t.CopyFrom(numpy_helper.from_array(np.array(-1.1920929e-07, dtype=np.float32), name=c20_name))
            c20_patched = True
            break
    if not c20_patched:
        for n in m.graph.node:
            if n.op_type == "Constant" and len(n.output) and n.output[0] == c20_name:
                n.attribute[0].t.CopyFrom(numpy_helper.from_array(np.array(-1.1920929e-07, dtype=np.float32)))
                c20_patched = True
                break

    onnx.save(m, str(dst_path))


def _try_download_from_hf(name):
    """Attempt downloading a model or voicepack from the user's Hugging Face repository."""
    from pdf_reader.models import MODEL_REPO, MODEL_REVISION
    try:
        from huggingface_hub import hf_hub_download
        rel = f"kokoro/{name}"
        print(f"Checking for {rel} in {MODEL_REPO} on Hugging Face...", flush=True)
        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=rel,
            revision=MODEL_REVISION,
            local_dir=MODELS_DIR,
        )
        p = pathlib.Path(downloaded)
        if p.exists() and p.stat().st_size > 0:
            print(f"Successfully downloaded {name} from {MODEL_REPO} on Hugging Face!")
            return True
    except Exception:
        pass
    return False


def ensure_files(model):
    """Download the model and voices file if needed; supports Hugging Face updates with local fallback."""
    if model in ("kokoro-v1.0.extreme.onnx", "kokoro-v1.0.fused_snake.onnx"):
        fused_path = KOKORO_DIR / model
        dll_path = SNAKE_DLL if SNAKE_DLL.exists() else pathlib.Path("tools/snake_op.dll")
        if fused_path.exists() and model in SHA256 and sha256(fused_path) != SHA256[model]:
            print(f"Notice: replacing outdated/invalid {model}...", flush=True)
            fused_path.unlink()
        if not fused_path.exists():
            if not _try_download_from_hf(model) or (model in SHA256 and sha256(fused_path) != SHA256[model]):
                print(f"Notice: {model} not available on Hugging Face; using fast model fallback...", flush=True)
                if fused_path.exists():
                    fused_path.unlink()
                return ensure_files("kokoro-v1.0.fast.onnx")
        if not dll_path.exists() or (dll_path.exists() and "snake_op.dll" in SHA256 and sha256(dll_path) != SHA256["snake_op.dll"]):
            if _try_download_from_hf("snake_op.dll"):
                downloaded_dll = KOKORO_DIR / "snake_op.dll"
                if downloaded_dll.exists():
                    dll_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(downloaded_dll, dll_path)
            if not dll_path.exists():
                print("Notice: tools/snake_op.dll not found; using portable fast model.", flush=True)
                return ensure_files("kokoro-v1.0.fast.onnx")
        ensure_files(VOICES_FILE)
        return model

    if model == "kokoro-v1.0.fast.onnx":
        fast_path = KOKORO_DIR / model
        if fast_path.exists() and "kokoro-v1.0.fast.onnx" in SHA256 and sha256(fast_path) != SHA256["kokoro-v1.0.fast.onnx"]:
            print("Notice: replacing outdated/distorted kokoro-v1.0.fast.onnx with fixed version...", flush=True)
            fast_path.unlink()
        if not fast_path.exists():
            if _try_download_from_hf(model):
                if fast_path.exists() and sha256(fast_path) != SHA256["kokoro-v1.0.fast.onnx"]:
                    print("Notice: downloaded kokoro-v1.0.fast.onnx from Hugging Face is the older distorted build; generating fixed version locally...", flush=True)
                    fast_path.unlink()
            if not fast_path.exists():
                ensure_files("kokoro-v1.0.onnx")
                ensure_files(VOICES_FILE)
                try:
                    print("Optimizing kokoro-v1.0.onnx (replacing STFT with fast Conv1D and DC phase fix)...", flush=True)
                    _patch_kokoro_stft(KOKORO_DIR / "kokoro-v1.0.onnx", fast_path)
                except Exception as e:
                    print(f"Warning: could not patch STFT ({e}); falling back to kokoro-v1.0.onnx", flush=True)
                    return "kokoro-v1.0.onnx"
        ensure_files(VOICES_FILE)
        return model

    for name in (model, VOICES_FILE):
        path = KOKORO_DIR / name
        if path.exists() and name in SHA256 and sha256(path) != SHA256[name]:
            print(f"Notice: replacing outdated/invalid {name}...", flush=True)
            path.unlink()
        if not path.exists():
            # Try Hugging Face first
            if not _try_download_from_hf(name):
                print(f"downloading {name} (Kokoro upstream release, once)", flush=True)
                _download(RELEASE_URL + name, path)
            if name in SHA256 and sha256(path) != SHA256[name]:
                path.unlink()
                raise RuntimeError(f"{name} did not verify (different from the tested file); removed")
    return model


def split_clauses(text, first_chunk_max=48, other_chunk_max=80):
    """Split long text into natural clauses and syntactic phrases.
    The first chunk is targeted at first_chunk_max (~48 chars, 6-8 words) so
    CPU neural synthesis finishes in ~1.0-1.1s for immediate playback on click.
    Subsequent chunks use other_chunk_max (~80 chars) for smooth, natural phrasing.
    Splits prefer:
    1. Natural punctuation marks (, ; : — –)
    2. Conjunctions and prepositions (that, which, using, where, with, and, for...)
    3. Word boundaries
    """
    if len(text) <= first_chunk_max:
        return [text]
    words = text.split(" ")
    chunks = []
    cur = []
    cur_len = 0
    limit = first_chunk_max

    punc_re = re.compile(r"[,;:\u2014\u2013]$")
    prep_set = {
        "that", "which", "using", "where", "when", "while", "with", "without",
        "and", "but", "or", "for", "by", "from", "into", "through", "during",
        "including", "thereby", "because", "since", "although", "whereas"
    }

    for w in words:
        w_len = len(w) + (1 if cur else 0)
        should_split = False
        if cur:
            if punc_re.search(cur[-1]) and (cur_len >= 20 or cur_len + w_len > limit):
                should_split = True
            elif (cur_len + w_len > limit * 0.7 or cur_len >= 32) and w.lower() in prep_set:
                should_split = True
            elif cur_len + w_len > limit:
                should_split = True

        if should_split and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
            limit = other_chunk_max
        else:
            cur.append(w)
            cur_len += w_len

    if cur:
        chunks.append(" ".join(cur))
    return chunks


class KokoroNarrator:
    aligned = False  # no phoneme durations: word timing is estimated

    def __init__(self, voice=None, speed=1.0):
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            raise SystemExit("The Kokoro voice needs one extra package: uv pip install kokoro-onnx") from None
        dll_path = SNAKE_DLL if SNAKE_DLL.exists() else pathlib.Path("tools/snake_op.dll")
        default_model = os.environ.get("PAPERVOICE_KOKORO_MODEL", "kokoro-v1.0.extreme.onnx")
        model = ensure_files(default_model)
        self.model_name = model

        import onnxruntime as ort
        cpus = os.cpu_count() or 4
        default_threads = min(6, cpus) if cpus >= 6 else cpus
        threads = int(os.environ.get("PAPERVOICE_KOKORO_THREADS", str(default_threads)))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if model in ("kokoro-v1.0.extreme.onnx", "kokoro-v1.0.fused_snake.onnx") and dll_path.exists():
            so.register_custom_ops_library(str(dll_path.resolve()))
        sess = ort.InferenceSession(str(KOKORO_DIR / model), so, providers=['CPUExecutionProvider'])
        self.kokoro = Kokoro.from_session(sess, str(KOKORO_DIR / VOICES_FILE))

        # Runtime preemption support for low-latency paragraph skipping & prefetch cancellation
        self._active_ro = None
        self._aborted = threading.Event()
        self._ro_lock = threading.Lock()
        orig_run = self.kokoro.sess.run

        def run_with_abort(*args, **kwargs):
            if self._aborted.is_set():
                raise RuntimeError("Kokoro inference aborted before execution")
            ro = ort.RunOptions()
            with self._ro_lock:
                self._active_ro = ro
            kwargs["run_options"] = ro
            try:
                if self._aborted.is_set():
                    ro.terminate = True
                return orig_run(*args, **kwargs)
            finally:
                with self._ro_lock:
                    if self._active_ro is ro:
                        self._active_ro = None

        self.kokoro.sess.run = run_with_abort
        self.voice_name = voice or os.environ.get("PAPERVOICE_KOKORO_VOICE", "af_heart")
        if self.voice_name not in self.kokoro.get_voices():
            raise SystemExit(f"Kokoro has no voice {self.voice_name!r}; choose one of: "
                             f"{', '.join(self.kokoro.get_voices())}")
        self.rate = 24000
        self.speed = speed

    def abort(self):
        """Preempt any active Kokoro inference mid-flight so high-priority requests can run."""
        self._aborted.set()
        with self._ro_lock:
            if self._active_ro is not None:
                self._active_ro.terminate = True

    def reset_abort(self):
        """Reset the abort state before running high-priority inference."""
        self._aborted.clear()

    @property
    def name(self):
        return self.voice_name

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        """1.0 = the voice's own pace, as for Piper (Kokoro's own speed parameter has the same meaning)."""
        if not 0.5 <= value <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        self._speed = value

    def silence(self, seconds):
        return b"\x00\x00" * int(self.rate * seconds / self.speed)

    def _pcm(self, text, pace):
        # math (pace < 1) is read slower than the set speed; Kokoro accepts 0.5 to 2.0
        speed = min(2.0, max(0.5, self._speed * pace))
        samples, _ = self.kokoro.create(text, voice=self.voice_name, speed=speed, lang="en-us")
        return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def sentences(self, text, pace=1.0):
        """PCM16 bytes. Natural sentence prosody; splits long sentences into natural clauses for sub-second latency."""
        if self._aborted.is_set():
            raise RuntimeError("Kokoro inference aborted")
        first_max = int(os.environ.get("PAPERVOICE_KOKORO_FIRST_CHUNK_CHARS", "48"))
        other_max = int(os.environ.get("PAPERVOICE_KOKORO_MAX_CHARS", "80"))
        parts = split_clauses(text, first_chunk_max=first_max, other_chunk_max=other_max)
        for part in parts:
            if self._aborted.is_set():
                raise RuntimeError("Kokoro inference aborted")
            yield self._pcm(part, pace)

    def chunks(self, text, pace=1.0):
        """[(letters, PCM16 bytes, None)]. Synthesizes sentences with complete natural acoustic context."""
        if self._aborted.is_set():
            raise RuntimeError("Kokoro inference aborted")
        first_max = int(os.environ.get("PAPERVOICE_KOKORO_FIRST_CHUNK_CHARS", "48"))
        other_max = int(os.environ.get("PAPERVOICE_KOKORO_MAX_CHARS", "80"))
        parts = split_clauses(text, first_chunk_max=first_max, other_chunk_max=other_max)
        out = []
        for part in parts:
            if self._aborted.is_set():
                raise RuntimeError("Kokoro inference aborted")
            out.append((list(part), self._pcm(part, pace), None))
        return out

    def chunk(self, text, chunk_idx, pace=1.0):
        """Synthesize only a single clause/chunk. Returns ([(letters, pcm, None)], total_chunks, part_text)."""
        if self._aborted.is_set():
            raise RuntimeError("Kokoro inference aborted")
        first_max = int(os.environ.get("PAPERVOICE_KOKORO_FIRST_CHUNK_CHARS", "48"))
        other_max = int(os.environ.get("PAPERVOICE_KOKORO_MAX_CHARS", "80"))
        parts = split_clauses(text, first_chunk_max=first_max, other_chunk_max=other_max)
        total = len(parts)
        if chunk_idx is None or chunk_idx < 0 or chunk_idx >= total:
            return self.chunks(text, pace), total, text
        part = parts[chunk_idx]
        return [(list(part), self._pcm(part, pace), None)], total, part

