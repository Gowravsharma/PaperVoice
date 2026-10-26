"""
check_highlight.py - replay the viewer's sentence/word locating on every sentence of an indexed PDF.

Usage: uv run python bench/check_highlight.py out/cache/<doc id> [--show 10]

The viewer finds a sentence's printed words in the page's text layer (viewer.js: locateSentence). This runs the
same steps in Python, with the PDF's own characters inside each block's box standing in for PDF.js's text layer,
and checks every highlighted word: it is right only if the letters under it are that word. Reports, for the
old method ("contiguous": find the whole sentence, else anchor on its last/first 24 letters, then assume the
words follow without gaps) and the new one ("anchored", as viewer.js now):
  - sentences lit in the wrong place (fewer than 80 % of their lit words right)
  - the share of lit words that are right
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pypdfium2 as pdfium


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKC", s).lower() if c.isalnum())


def block_letters(page, bbox, height):
    """letters of the text layer inside the block box (PDF points, top-left origin), in content order"""
    tp = page.get_textpage()
    l, t, r, b = bbox
    out = []
    for i in range(tp.count_chars()):
        ch = tp.get_text_range(i, 1)
        if not ch or not ch.strip():
            continue
        x0, y0, x1, y1 = tp.get_charbox(i)
        cx, cy = (x0 + x1) / 2, height - (y0 + y1) / 2
        if l - 3 <= cx <= r + 3 and t - 3 <= cy <= b + 3:
            out.extend(c for c in unicodedata.normalize("NFKC", ch).lower() if c.isalnum())
    return "".join(out)


def contiguous(flat, text):
    """viewer.js locateSentence before the fix"""
    want = norm(text)
    if not want or not flat:
        return None
    start = flat.find(want)
    if start < 0:
        tail, head = want[-24:], want[:24]
        ts, hs = flat.find(tail), flat.find(head)
        if ts >= 0:
            start = max(0, ts + len(tail) - len(want))
        elif hs >= 0:
            start = hs
        else:
            return None
    words, off = [], start
    for w in text.split():
        n = len(norm(w))
        words.append((min(off, len(flat)), min(off + n, len(flat))))
        off += n
    return words


def anchored(flat, text, start_at=0):
    """the fix, as in viewer.js locateSentence: start after the previous sentence of the block (start_at) where
    three of its words follow each other (then two, then one long word; words before that stay unlit: None); then
    each word is looked for near where the previous one ended, so a symbol only one text has cannot shift the
    rest; a word not found close by is left unlit (as viewer.js)"""
    ws = [norm(w) for w in text.split()]

    def follows(k, at, upto):  # how many of words k.. appear one after another from letter `at`
        ok = skipped = 0
        for i in range(k, min(len(ws), k + upto)):
            if not ws[i]:
                continue
            j = flat.find(ws[i], at)
            if j < 0 or j > at + 3:
                if len(ws[i]) <= 2 and skipped < 2:  # a maths letter or sub/superscript printed elsewhere
                    skipped += 1
                    ok += 1
                    continue
                break
            ok += 1
            at = j + len(ws[i])
        return ok

    pos, first, best = -1, 0, None
    tiny = len(ws) > 3 and len("".join(ws)) < 8  # only tiny words: looked for whole (as viewer.js)
    for n in ((len(ws), 3, 2, 1) if tiny else (3, 2, 1)):  # a verified run: three words, then two, then one long word
        for k in range(0, len(ws) - n + 1):
            key = "".join(ws[k:k + n])
            if len(key) < (5 if n == 1 else 4) and not (k == 0 and n == len(ws)):
                continue
            need = sum(1 for w in ws[k:k + 8] if w)  # words with letters: follows() skips symbols (as viewer.js)
            for start in (start_at, 0):
                j = flat.find(key, start)
                while j >= 0:
                    ok = follows(k, j, 8)
                    if ok >= need:
                        pos, first = j, k
                        break
                    if best is None or ok > best[0]:
                        best = (ok, j, k)
                    j = flat.find(key, j + 1)
                if pos >= 0 or start == 0:
                    break
            if pos >= 0:
                break
        if pos >= 0:
            break
    if pos < 0 and best and best[0] >= 3:
        _, pos, first = best
    if pos < 0:
        return None
    words = [None] * len(ws)
    for k in range(first, len(ws)):
        w = ws[k]
        if not w:
            continue
        j = flat.find(w, pos)
        if j < 0 or j > pos + 40:
            continue
        words[k] = (j, j + len(w))
        pos = j + len(w)
    return words


def truth(flat, text, start_hint):
    """where each word really is: greedy exact search from the sentence's true start (for scoring only)"""
    return anchored(flat, text, start_hint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    folder = Path(a.cache)
    pdf = pdfium.PdfDocument(str(folder / "document.pdf"))
    # per method: sentences, lit words, lit words whose letters really are that word, sentences with a wrong word
    stats = {m: {"sentences": 0, "lit": 0, "right": 0, "bad_sentences": 0, "not_found": 0} for m in ("old", "new")}
    shown = 0
    for pf in sorted(folder.glob("page_*.json")):
        data = json.loads(pf.read_text(encoding="utf-8"))
        p = data["number"]
        page = pdf[p - 1]
        height = page.get_height()
        for b in data["blocks"]:
            if b["kind"] not in ("text", "caption", "heading") or not b.get("bbox"):
                continue
            flat = block_letters(page, b["bbox"], height)
            cursor = 0
            for s in b["sentences"]:
                text = s["text"]
                ws = [norm(w) for w in text.split()]
                for name, res in (("old", contiguous(flat, text)), ("new", anchored(flat, text, cursor))):
                    st = stats[name]
                    st["sentences"] += 1
                    if res is None:
                        st["not_found"] += 1
                        continue
                    lit = [(k, r) for k, r in enumerate(res) if r and ws[k]]
                    right = sum(flat[r[0]:r[1]] == ws[k] for k, r in lit)
                    st["lit"] += len(lit)
                    st["right"] += right
                    if lit and right < 0.8 * len(lit):
                        st["bad_sentences"] += 1
                        if name == "new" and shown < a.show:
                            shown += 1
                            print(f"p{p} {s['id']}: {right}/{len(lit)} words right: {text[:80]!r}")
                    if name == "new":
                        ends = [r[1] for r in res if r]
                        cursor = ends[-1] if ends else cursor
    for name, st in stats.items():
        print(f"{name}: {st['sentences']} sentences, {st['bad_sentences']} lit in the wrong place (<80 % of words "
              f"right), {st['not_found']} not found; lit words right {st['right']}/{st['lit']} "
              f"({100 * st['right'] / max(1, st['lit']):.1f} %)")


if __name__ == "__main__":
    main()
