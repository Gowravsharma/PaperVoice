"""
speak_tables.py - listen to the tables of a page alone: their units from an audit run, re-spoken with the
current rules and the planned pauses.

Usage: uv run python bench/speak_tables.py out/audit/<run>/<paper>_p7-7/05_piper_chunks.json out/samples/tables.wav
(The audit stores the spoken text; text rules changed since the audit only apply after re-running it.)
"""
import json
import sys
import wave
from pathlib import Path

from pdf_reader.narration import PAUSES
from pdf_reader.read_aloud import Narrator


def main():
    units = [u for u in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")) if u["kind"] == "table"]
    n = Narrator()
    pcm = []
    for u in units:
        pcm += list(n.sentences(u["text_to_piper"], u.get("pace", 1.0))) + [n.silence(PAUSES[u["pause_after"]])]
    with wave.open(sys.argv[2], "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(n.rate)
        w.writeframes(b"".join(pcm))
    print(f"{len(units)} table sentences -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
