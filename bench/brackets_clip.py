import os, sys, wave
from pdf_reader.read_aloud import make_narrator
from pdf_reader.text_norm import prose, symbols, parentheticals
engine = sys.argv[1]
os.environ["PAPERVOICE_VOICE_ENGINE"] = engine
n = make_narrator()
S = ["We distil a large teacher model (i.e., knowledge distillation) into a smaller student that is much faster.",
     "Automatic speech recognition (ASR) systems often fail on rare words (see Section 2), but they are improving.",
     "The duration predictor is trained on the alignments (extracted from an external aligner) and then used at inference."]
for tag, f in (("before", lambda s: symbols(prose(s))), ("after", lambda s: parentheticals(symbols(prose(s))))):
    pcm, secs = b"", []
    for s in S:
        a = b"".join(c[1] for c in n.chunks(f(s), 1.0))
        secs.append(len(a) / 2 / n.rate); pcm += a + n.silence(0.7)
    with wave.open(f"out/listen/brackets/{engine}_{tag}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(n.rate); w.writeframes(pcm)
    print(engine, tag, [round(x, 2) for x in secs])
