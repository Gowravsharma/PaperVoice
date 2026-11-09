import urllib.request
import json
import wave
import io

# 1. Check health
health = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/health").read())
print("Server Health:", health)
assert health.get("engine") == "kokoro", f"Expected kokoro, got {health.get('engine')}"
assert health.get("voice") == "af_heart", f"Expected af_heart, got {health.get('voice')}"

# 2. Open a test PDF
with open("test_papers/p5.pdf", "rb") as f:
    pdf_bytes = f.read()

req = urllib.request.Request("http://127.0.0.1:8765/api/open?name=p5.pdf", data=pdf_bytes, headers={"Content-Type": "application/pdf"})
res = json.loads(urllib.request.urlopen(req).read())
print("Opened Doc:", res)
doc_id = res["doc"]

# 3. Request page 1 blocks to get the first sentence
page_info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8765/api/doc/{doc_id}/page/1?wait=1").read())
first_sid = page_info["blocks"][0]["sentences"][0]["id"]
print("First Sentence ID:", first_sid)

# 4. Request audio for the first sentence
audio_bytes = urllib.request.urlopen(f"http://127.0.0.1:8765/api/doc/{doc_id}/audio/{first_sid}?speed=1.0").read()
with wave.open(io.BytesIO(audio_bytes), "rb") as w:
    channels = w.getnchannels()
    sampwidth = w.getsampwidth()
    framerate = w.getframerate()
    frames = w.getnframes()
    duration = frames / framerate
    print(f"Audio received: {len(audio_bytes)} bytes | Rate: {framerate} Hz | Channels: {channels} | Width: {sampwidth} | Duration: {duration:.2f} s")
    assert framerate == 24000, f"Expected 24000 Hz Kokoro audio, got {framerate}"

# 5. Request timing for the first sentence
timing_info = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8765/api/doc/{doc_id}/timing/{first_sid}?speed=1.0").read())
print(f"Timing received: {len(timing_info['words'])} words, duration: {timing_info['duration']:.2f} s")
print("SUCCESS: Kokoro is actively serving audio to the extension and viewer!")
