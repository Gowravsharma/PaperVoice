import os, sys, time, statistics
os.environ["PAPERVOICE_VOICE_ENGINE"] = sys.argv[1]
from pdf_reader.read_aloud import make_narrator
n = make_narrator()
text = "The model reads research papers aloud, and the listener follows each sentence on the page as it is spoken."
n.chunks(text, 1.0)  # warm-up
times, audio = [], 0
for _ in range(5):
    t = time.perf_counter(); c = n.chunks(text, 1.0); times.append(time.perf_counter() - t)
    audio = sum(len(x[1]) for x in c) / 2 / n.rate
print(f"{sys.argv[1]:7} {len(text)} chars -> {audio:.2f} s of audio; compute {statistics.median(times):.2f} s (median of 5; min {min(times):.2f}, max {max(times):.2f}); real-time factor {statistics.median(times)/audio:.2f}")
