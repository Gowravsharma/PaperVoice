"""
check_numbers.py - how a bare "(2.1)" in the running text is read, with the live map of the paper's numbers.

Usage: uv run python bench/check_numbers.py paper.pdf [--pages 1-12] [--show 12]

Runs the real streaming pipeline (PageStreamer) over the pages, in order, so the map fills page by page exactly as
during reading (script.note_numbers). Every "(d.d)" that text_norm.prose meets in the index step is recorded with
what the map knew at that moment and how it was read:
  equation N / section N   the map had exactly that kind
  N alone                  both kinds, or the number is not in the map (yet)
  N alone (value)          starts with "0." - a weight, probability or standard error
Then the same pages are loaded from the page cache (written by the first run) into a new streamer: its map must
be the same as the first run's, proving a cached page adds what it defines.
"""
import argparse
import re
import tempfile
from collections import Counter

from pdf_reader import index, text_norm
from pdf_reader.stream import PageStreamer, open_document, page_count, parse_pages


def run(pdf, pages, cache_dir):
    seen = []
    original, in_index = text_norm._number_reference, [False]
    inner = index._speak_text

    def spy(num, refs):
        out = original(num, refs)
        if in_index[0]:
            kinds = None if refs is None else sorted(refs.get(num, ()))
            seen.append((num, kinds, out))
        return out

    def wrapped(sentence, refs=None):
        in_index[0] = True
        try:
            return inner(sentence, refs)
        finally:
            in_index[0] = False

    text_norm._number_reference, index._speak_text = spy, wrapped
    try:
        s = PageStreamer(pdf, pages=pages, cache_dir=cache_dir)
        for p in pages:
            s.get(p)  # the listener arrives at page p: the worker prepares it and the pages after it
        return s, seen
    finally:
        text_norm._number_reference, index._speak_text = original, inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--pages", default=None, help="e.g. 1-12 (default: all)")
    ap.add_argument("--show", type=int, default=8, help="examples to print")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as cache:
        n_pages = page_count(open_document(a.pdf))
        pages = parse_pages(a.pages, n_pages)
        s, seen = run(a.pdf, pages, cache)
        kinds = Counter(k for ks in s.refs.values() for k in ks)
        print(f"{a.pdf}: {len(pages)} pages; map: {len(s.refs)} numbers ({dict(kinds)}); "
              f"{sum(1 for ks in s.refs.values() if len(ks) > 1)} carried by both kinds")
        how = Counter()
        for num, ks, out in seen:
            how["value (0.x)" if num.startswith("0.") else out.split(" ")[0] if " " in out else "number alone"] += 1
        print(f"bare (d.d) read: {len(seen)} -> {dict(how)}")
        for num, ks, out in seen[:a.show]:
            print(f"   ({num})  map knew {ks}  ->  '{out}'")
        again = PageStreamer(a.pdf, pages=pages, cache_dir=cache)
        same = {n: sorted(k) for n, k in again.refs.items()} == {n: sorted(k) for n, k in s.refs.items()}
        print("map restored from the page cache is identical:", same)


if __name__ == "__main__":
    main()
