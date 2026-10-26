"""
read_aloud.py - the full chain: PDF -> streamed pages -> Piper voice -> speakers and/or WAV files.

    uv run papervoice-read paper.pdf --play                      # whole paper on the speakers
    uv run papervoice-read paper.pdf --start 12 --play           # from page 12 to the end
    uv run papervoice-read paper.pdf --pages 2,5,9-11 --save out/audio
    options: --no-page-numbers (no spoken "Page five."), --voice en_US-lessac-medium
Each page starts with a spoken "Page N." and a console line "▶ page N of M", so it is easy to tell which
page is being heard when testing.

Three stages run at the same time, each feeding the next through a queue:
  1. PageStreamer (stream.py)  prepares pages in the background (layout, formulas, tables, normalisation)
  2. voice thread              Piper synthesises sentence by sentence (~17-20x faster than real time here)
  3. output                    --play: Windows speakers (winsound); --save: one WAV per page
So the first sentence is heard a few seconds after page 1 is ready, not after the whole page or paper.

Each speech unit (sentence, formula, heading) is one Piper call followed by its planned pause (narration.py:
clause / sentence / paragraph / section), plus PAGE_PAUSE after a page. Voices live in models/voices/
(download: python -m piper.download_voices --download-dir models/voices <name>).
"""
import argparse
import io
import queue
import threading
import time
import wave
from pathlib import Path

from piper import PiperVoice, SynthesisConfig

from pdf_reader.models import VOICES_DIR
from pdf_reader.narration import PAUSES, plan_page
from pdf_reader.stream import PageStreamer, parse_pages

PAGE_PAUSE = 0.5  # added after a page, on top of the last unit's own pause (narration.PAUSES)
END = object()


def aligned_voice(voice):
    """<voice>.aligned.onnx: the voice with its per-phoneme durations as a second output (exact word timings,
    timing.word_times_aligned). Made once from the downloaded voice with Piper's patch_voice_with_alignment
    (needs the onnx package, installed with piper-tts[alignment]); None if that is not possible."""
    src, dst = VOICES_DIR / f"{voice}.onnx", VOICES_DIR / f"{voice}.aligned.onnx"
    if not dst.exists():
        try:
            import onnx
            from piper.patch_voice_with_alignment import add_alignment_output
            model = onnx.load(str(src))
            add_alignment_output(model)
            tmp = dst.with_suffix(".part")
            onnx.save(model, str(tmp))
            tmp.replace(dst)
        except (ImportError, ValueError):
            return None
    return dst


class Narrator:
    def __init__(self, voice="en_US-lessac-medium", speed=1.0):
        aligned = aligned_voice(voice)
        config = VOICES_DIR / f"{voice}.onnx.json"
        self.voice = PiperVoice.load(str(aligned or VOICES_DIR / f"{voice}.onnx"), config_path=str(config))
        self.aligned = aligned is not None
        self.rate = self.voice.config.sample_rate
        self.speed = speed

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        """1.0 = the voice's own pace; 0.8 = 20 % slower, 1.25 = 25 % faster. Piper stretches the phonemes
        (length_scale = 1 / speed), so the pitch stays natural. Takes effect from the next sentence."""
        if not 0.5 <= value <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        self._speed = value
        self.config = SynthesisConfig(length_scale=1.0 / value)

    def silence(self, seconds):
        return b"\x00\x00" * int(self.rate * seconds / self.speed)

    def config_for(self, pace=1.0):
        return self.config if pace == 1.0 else SynthesisConfig(length_scale=1.0 / (self._speed * pace))

    def sentences(self, text, pace=1.0):
        """PCM16 bytes, one item per sentence. pace < 1 reads this text slower than the set speed (math)."""
        for chunk in self.voice.synthesize(text, syn_config=self.config_for(pace)):
            yield chunk.audio_int16_bytes

    def chunks(self, text, pace=1.0):
        """[(phonemes, PCM16 bytes, alignments or None)] - alignments: [(phoneme, samples)] when the voice
        outputs its durations (see aligned_voice)."""
        out = []
        for c in self.voice.synthesize(text, syn_config=self.config_for(pace), include_alignments=self.aligned):
            al = [(a.phoneme, int(a.num_samples)) for a in c.phoneme_alignments] if c.phoneme_alignments else None
            out.append((c.phonemes, c.audio_int16_bytes, al))
        return out


def wav_bytes(pcm, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class ReadAloud:
    """Runs the three stages; `log` collects timings for the report."""

    def __init__(self, pdf, pages, narrator, streamer, play=False, save_dir=None, buffer_sentences=30,
                 announce=True):
        self.pages, self.narrator, self.streamer = pages, narrator, streamer
        self.announce = announce  # say "Page five." at the start of each page (easy to tell where you are)
        self.play, self.save_dir = play, Path(save_dir) if save_dir else None
        self.audio = queue.Queue(maxsize=buffer_sentences)  # bounded: synthesis waits if far ahead of output
        self.t0 = streamer.t0
        self.log = {"first_audio": None, "underruns": 0, "underrun_seconds": 0.0, "audio_seconds": 0.0,
                    "synth_seconds": 0.0, "pages": {}}
        self.stem = Path(pdf).stem

    # stage 2
    def _synthesize(self):
        rate = self.narrator.rate
        for i, p in enumerate(self.pages):
            page = self.streamer.get(p)
            self.audio.put(("page", p))
            tail = None
            if page.tail:
                # the sentence cut by the page break: if the next page is read next, wait for it - it takes the
                # sentence over when it is prepared right after this one; otherwise say the fragment here
                nxt = self.pages[i + 1] if i + 1 < len(self.pages) else None
                if nxt == p + 1:
                    self.streamer.get(nxt)
                if not page.tail_joined:
                    tail = page.tail["sentence"]
            # one Piper call per speech unit (sentence, formula, heading), then the planned pause (narration.py)
            units = plan_page(page.blocks, p, tail, announce=self.announce)
            if not units:
                continue
            for u in units:
                t = time.perf_counter()
                for pcm in self.narrator.sentences(u.text, u.pace):
                    self.log["synth_seconds"] += time.perf_counter() - t
                    self.audio.put(("pcm", pcm))
                    t = time.perf_counter()
                self.audio.put(("pcm", self.narrator.silence(PAUSES[u.pause])))
            self.audio.put(("pcm", self.narrator.silence(PAGE_PAUSE)))
        self.audio.put((END, None))

    # stage 3
    def run(self):
        threading.Thread(target=self._synthesize, daemon=True).start()
        rate, writer, page = self.narrator.rate, None, None
        played_until = None  # when the speakers will be free (play mode)
        while True:
            t_wait = time.perf_counter()
            kind, item = self.audio.get()
            waited = time.perf_counter() - t_wait
            if kind is END:
                break
            if kind == "page":
                if writer:
                    writer.close()
                page = item
                if self.save_dir:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    writer = wave.open(str(self.save_dir / f"{self.stem}_p{page:03d}.wav"), "wb")
                    writer.setnchannels(1)
                    writer.setsampwidth(2)
                    writer.setframerate(rate)
                self.log["pages"][page] = {"first_audio": None, "seconds": 0.0}
                continue
            seconds = len(item) / 2 / rate
            now = time.perf_counter() - self.t0
            if self.log["first_audio"] is None:
                self.log["first_audio"] = now
            if self.log["pages"][page]["first_audio"] is None:
                self.log["pages"][page]["first_audio"] = now
                print(f"▶ page {page} of {self.streamer.n_pages}" + (" (playing)" if self.play else ""), flush=True)
            self.log["pages"][page]["seconds"] += seconds
            # an underrun is the listener hearing silence because the next sentence was not ready in time;
            # in save-only mode this is computed against a virtual playback clock
            clock = played_until if played_until is not None else now
            if played_until is not None and now > played_until + 0.05:
                self.log["underruns"] += 1
                self.log["underrun_seconds"] += now - played_until
                clock = now
            played_until = clock + seconds
            self.log["audio_seconds"] += seconds
            if writer:
                writer.writeframes(item)
            if self.play:
                import winsound
                winsound.PlaySound(wav_bytes(item, rate), winsound.SND_MEMORY)  # blocks while it plays
                played_until = time.perf_counter() - self.t0
        if writer:
            writer.close()
        return self.log


def main():
    import psutil
    # module imports (Docling, PyTorch, Piper) happen before main() runs; count them in "time to first sound"
    startup, main_start = time.time() - psutil.Process().create_time(), time.perf_counter()
    ap = argparse.ArgumentParser(description="Read a PDF aloud: streamed pages -> Piper voice")
    ap.add_argument("pdf")
    ap.add_argument("--pages", help='pages to read: "5", "3-7", "2,5,9-11", "7-" (default: all)')
    ap.add_argument("--start", type=int, help="start at this page and read to the end")
    ap.add_argument("--no-page-numbers", action="store_true", help='do not say "Page five." at each page')
    ap.add_argument("--play", action="store_true", help="play on this computer's speakers")
    ap.add_argument("--save", help="directory for one WAV file per page")
    ap.add_argument("--voice", default="en_US-lessac-medium")
    ap.add_argument("--speed", type=float, default=0.8,
                    help="speaking speed: 1.0 normal, 0.8 slower, 1.2 faster (0.5-2.0)")
    ap.add_argument("--prefetch", type=int, default=3)
    a = ap.parse_args()
    if not (a.play or a.save):
        ap.error("choose --play and/or --save DIR")
    from pdf_reader.pdfium_safe import open_document, page_count
    try:
        pages = parse_pages(a.pages, page_count(open_document(a.pdf)), a.start)
    except ValueError as e:
        ap.error(str(e))

    from pdf_reader.models import ensure
    ensure()  # first start of a fresh install: downloads the models
    streamer = PageStreamer(a.pdf, prefetch=a.prefetch, pages=pages)
    try:
        narrator = Narrator(a.voice, a.speed)  # loads while the page worker loads its models
    except ValueError as e:
        ap.error(str(e))
    log = ReadAloud(a.pdf, pages, narrator, streamer, a.play, a.save, announce=not a.no_page_numbers).run()

    since_launch = startup + (streamer.t0 - main_start)
    print(f"Python start-up and imports before the pipeline started: {since_launch:.1f}s")
    print(f"pages {a.pages or (str(a.start) + '-' if a.start else 'all')}: first sound after {log['first_audio']:.1f}s "
          f"({log['first_audio'] + since_launch:.1f}s after launch); "
          f"{log['audio_seconds'] / 60:.1f} min of audio synthesised in {log['synth_seconds']:.0f}s "
          f"(real-time factor {log['synth_seconds'] / max(log['audio_seconds'], 1e-9):.2f}); "
          f"listener heard a gap {log['underruns']} times ({log['underrun_seconds']:.1f}s)")
    for p, info in log["pages"].items():
        print(f"  page {p:3d}: starts at {info['first_audio']:6.1f}s, {info['seconds'] / 60:4.1f} min of audio")
    if a.save:
        print(f"WAV files in {a.save}")
