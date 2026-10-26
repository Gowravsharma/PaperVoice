"""
audit_script.py - list everything in a TTS script that is not yet plain words, by kind.

Usage: uv run python bench/audit_script.py out/paper.speech.txt [more.speech.txt ...]
"""
import collections
import re
import sys

KINDS = [  # (name, pattern) - first match wins, so specific kinds come first
    ("citation [12] / [3, 4]", r"\[\d+(?:\s*[,–-]\s*\d+)*\]"),
    ("equation/section ref (2.1)", r"\(\d+\.\d+\)"),
    ("year 1990-2029", r"\b(?:19[5-9]\d|20[0-2]\d)\b"),
    ("number with thousands comma 6,139", r"\b\d{1,3}(?:,\d{3})+\b"),
    ("decimal 0.836", r"\b\d+\.\d+\b"),
    ("range or flattened power 10-1", r"\b\d+\s?[-–]\s?\d+\b"),
    ("digits glued to letters 1Zhejiang / Lp", r"\b\d+[A-Za-z]+\w*|\b[A-Za-z]+\d+\w*"),
    ("integer", r"\b\d+\b"),
    ("url / e-mail", r"https?://\S+|\S+@\S+"),
    ("abbreviation e.g. / i.e. / r.v.", r"\b(?:e\.g|i\.e|r\.v|et al|cf|vs|resp|w\.r\.t)\."),
]
SYMBOL = re.compile(r"[^\w\s.,;:!?'\"()\[\]{}\-–—/]")


def main():
    for path in sys.argv[1:]:
        text = open(path, encoding="utf-8").read()
        rest = text
        counts, examples = collections.Counter(), collections.defaultdict(list)
        for name, pat in KINDS:
            for m in re.finditer(pat, rest):
                counts[name] += 1
                if len(examples[name]) < 4 and m.group(0) not in examples[name]:
                    examples[name].append(m.group(0))
            rest = re.sub(pat, " ", rest)
        syms = collections.Counter(SYMBOL.findall(text))
        print(f"== {path}  ({len(text.split())} words)")
        for name, _ in KINDS:
            if counts[name]:
                print(f"  {counts[name]:5d}  {name:<38} e.g. {', '.join(examples[name])}")
        print(f"  {sum(syms.values()):5d}  symbols left: " + " ".join(f"{s}×{n}" for s, n in syms.most_common(18)))


if __name__ == "__main__":
    main()
