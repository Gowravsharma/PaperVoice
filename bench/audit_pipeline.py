"""
audit_pipeline.py - trace one passage through every stage and save what each stage produced.

Usage (main project env):
    uv run python bench/audit_pipeline.py test_papers/a_probabilistic_numeric_method.pdf --pages 5-5 \
        [--speed 1.0] [--out out/audit]

Writes to <out>/<pdf stem>_p<pages>/:
  01_docling_structure.json   Docling items in reading order: label, page, box, text (what the layout found)
  02_extracted_text.txt       the same as plain text, one block per paragraph, labelled
  03_math_sre_output.txt      per display formula: recognised LaTeX, cleaned LaTeX, the speech engine's own
                              (literal) reading, and our readable reading
  04_normalized_narration.txt the final spoken text, block by block and sentence by sentence
  05_piper_chunks.json        exactly what each Piper call received, and the sentence chunks it returned
  06_audio_metrics.json       measured from the audio: duration, words per minute, and every pause
  audio.wav                   the passage as papervoice-read would speak it (for listening)
"""
import argparse
import json
import re
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from docling.datamodel.base_models import InputFormat
from piper import SynthesisConfig
from pdf_reader.formula_onnx import make_converter
from pdf_reader.index import page_blocks
from pdf_reader.latex_cleanup import tidy
from pdf_reader.narration import PAUSES, plan_page
from pdf_reader.read_aloud import PAGE_PAUSE, Narrator
from pdf_reader.script import TEX2SPEECH, FormulaSpeaker, new_stats


def silences(pcm, rate, min_ms=40, rel_db=-35.0):
    """Silent stretches in 16-bit PCM: 10 ms frames quieter than rel_db below the loudest frame."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    hop = rate // 100
    n = len(x) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9)
    db = 20 * np.log10(rms / (rms.max() + 1e-9))
    quiet = db < rel_db
    out, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * 10 >= min_ms:
                out.append((start / 100, i / 100))
            start = None
    return out


def literal_speech(latex_list):
    """The speech engine's own reading (no readable.mjs rewrite)."""
    lines = ["$" + " ".join(l.split()) + "$" for l in latex_list if l.strip()]
    if not lines:
        return []
    r = subprocess.run(["node", str(TEX2SPEECH), "clearspeak", "--literal"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8")
    return r.stdout.rstrip("\n").split("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="5-5")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default="out/audit")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.pages.split("-"))
    out = Path(a.out) / f"{Path(a.pdf).stem}_p{a.pages}"
    out.mkdir(parents=True, exist_ok=True)

    conv = make_converter(4)
    conv.initialize_pipeline(InputFormat.PDF)
    speaker = FormulaSpeaker()
    narrator = Narrator(speed=a.speed)
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(a.pdf)

    structure, extracted, math_rows, narration, chunks_log = [], [], [], [], []
    pcm_all, timeline, clock = [], [], 0.0
    state = {"in_refs": False}
    for p in range(lo, hi + 1):
        doc = conv.convert(a.pdf, page_range=(p, p)).document
        for it, _ in doc.iterate_items():
            bb = it.prov[0].bbox.to_top_left_origin(doc.pages[p].size.height) if it.prov else None
            structure.append({"page": p, "label": str(it.label), "bbox": [round(v, 1) for v in (bb.l, bb.t, bb.r, bb.b)]
                              if bb else None, "text": getattr(it, "text", ""), "orig": getattr(it, "orig", "")})
            extracted.append(f"[p{p} {it.label}] {' '.join((getattr(it, 'text', '') or '').split())}")
        formulas = [it.text for it, _ in doc.iterate_items() if str(it.label) == "formula"]
        cleaned = [tidy(f)[0] for f in formulas]
        literal = literal_speech(cleaned)
        readable = speaker(cleaned)
        for raw, c, lit, rd in zip(formulas, cleaned, literal + [""] * len(cleaned), readable):
            math_rows.append(f"RAW      {raw}\nCLEANED  {c}\nLITERAL  {lit}\nREADABLE {rd}\n")

        blocks = page_blocks(doc, pdf, speaker, state, new_stats(), p)
        for b in blocks:
            narration.append(f"[{b.id} {b.kind}]")
            for s in b.sentences:
                narration.append(f"   printed: {s.text}\n   spoken : {s.spoken}")
        # what papervoice-read sends to Piper: one call per speech unit, then the planned pause (narration.py)
        for u in plan_page(blocks, p):
            text = u.text
            t = time.perf_counter()
            cfg = SynthesisConfig(length_scale=1 / (a.speed * u.pace))
            got = [(c.phonemes, c.audio_int16_bytes) for c in narrator.voice.synthesize(text, syn_config=cfg)]
            synth = time.perf_counter() - t
            entry = {"block": u.block, "sentence": u.sentence, "kind": u.kind, "pause_after": u.pause, "pace": u.pace,
                     "text_to_piper": text, "synth_seconds": round(synth, 3), "piper_chunks": []}
            for ph, audio in got:
                dur = len(audio) / 2 / narrator.rate
                sil = silences(audio, narrator.rate)
                entry["piper_chunks"].append({
                    "phonemes": "".join(ph), "seconds": round(dur, 3),
                    "leading_silence": round(sil[0][1], 3) if sil and sil[0][0] == 0 else 0.0,
                    "trailing_silence": round(dur - sil[-1][0], 3) if sil and abs(sil[-1][1] - dur) < 0.02 else 0.0,
                    "inner_pauses": [round(e - s, 3) for s, e in sil if s > 0 and abs(e - dur) >= 0.02]})
                pcm_all.append(audio)
                timeline.append(("chunk", u.block, clock, clock + dur))
                clock += dur
            chunks_log.append(entry)
            gap = narrator.silence(PAUSES[u.pause])
            pcm_all.append(gap)
            clock += len(gap) / 2 / narrator.rate
        gap = narrator.silence(PAGE_PAUSE)
        pcm_all.append(gap)
        clock += len(gap) / 2 / narrator.rate
    speaker.close()

    pcm = b"".join(pcm_all)
    with wave.open(str(out / "audio.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(narrator.rate)
        w.writeframes(pcm)

    # 06: pauses heard by the listener, classified by what is at that point in the text
    words = sum(len(e["text_to_piper"].split()) for e in chunks_log)
    speech_s = sum(c["seconds"] for e in chunks_log for c in e["piper_chunks"])
    # what the listener hears between consecutive units: Piper's own silence + the planned pause, per class
    heard = {}
    for e, nxt in zip(chunks_log, chunks_log[1:]):
        if not e["piper_chunks"] or not nxt["piper_chunks"]:
            continue
        gap = e["piper_chunks"][-1]["trailing_silence"] + PAUSES[e["pause_after"]] / a.speed + nxt["piper_chunks"][0]["leading_silence"]
        heard.setdefault(e["pause_after"], []).append(round(gap, 3))
    between_sentences = heard.get("sentence", [])
    inner = [p for e in chunks_log for c in e["piper_chunks"] for p in c["inner_pauses"]]
    commas = sum(c["phonemes"].count(",") for e in chunks_log for c in e["piper_chunks"])
    metrics = {
        "speed_setting": a.speed, "length_scale": round(1 / a.speed, 3),
        "total_audio_seconds": round(len(pcm) / 2 / narrator.rate, 2), "speech_seconds": round(speech_s, 2),
        "words": words, "words_per_minute_while_speaking": round(words / speech_s * 60, 1),
        "piper_calls": len(chunks_log), "piper_sentence_chunks": sum(len(e["piper_chunks"]) for e in chunks_log),
        "pause_between_sentences_in_a_paragraph_s": {
            "count": len(between_sentences), "median": float(np.median(between_sentences)) if between_sentences else None,
            "min": min(between_sentences, default=None), "max": max(between_sentences, default=None)},
        "pauses_inside_sentences_s": {"count": len(inner), "commas_in_text": commas,
                                      "median": float(np.median(inner)) if inner else None,
                                      "values": sorted(inner)[:40]},
        "heard_pause_by_class_s": {k: {"count": len(v), "median": float(np.median(v))} for k, v in heard.items()},
        "added_pause_between_pages_s": PAGE_PAUSE / a.speed,
        "synthesis_seconds": round(sum(e["synth_seconds"] for e in chunks_log), 2),
    }

    (out / "01_docling_structure.json").write_text(json.dumps(structure, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "02_extracted_text.txt").write_text("\n\n".join(extracted), encoding="utf-8")
    (out / "03_math_sre_output.txt").write_text("\n".join(math_rows), encoding="utf-8")
    (out / "04_normalized_narration.txt").write_text("\n".join(narration), encoding="utf-8")
    (out / "05_piper_chunks.json").write_text(json.dumps(chunks_log, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "06_audio_metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    print(f"artifacts in {out}")


if __name__ == "__main__":
    main()
