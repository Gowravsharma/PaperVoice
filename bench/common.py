"""Shared helpers for formula-recognizer benchmarks: crops, ground truth, LaTeX scoring."""
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# spelling variants that render / read the same; normalised before scoring
SUBS = [
    (r"\\mbox\s*\{", r"\\text{"), (r"\\rm\s*", ""), (r"\\limits", ""), (r"\\nolimits", ""),
    (r"\\mathop\s*\{\s*\\(\w+)\s*\}", r"\\\1"), (r"\\left\s*", ""), (r"\\right\s*", ""),
    (r"\\colon\s*=", ":="), (r"\\colon", ":"), (r"\\(?:,|;|:|!|quad|qquad)\b|~|\\ ", " "),
    (r"\\operatorname\s*\{\s*(\w+)\s*\}", r"\\\1"), (r"\\displaystyle", ""), (r"\\hat\s*\{\s*(\w)\s*\}", r"\\hat \1"),
    (r"\\begin\{array\}\{[^}]*\}|\\end\{array\}", ""), (r"\\(?:tag|label)\s*\{[^}]*\}", ""),
    (r"\\operatorname\*?\s*\{([a-z ]+)\}", lambda m: "\\" + m.group(1).replace(" ", "")),
    (r"\\mathrm\s*\{([a-z ~]+)\}", lambda m: r"\text{" + m.group(1) + "}"), (r"&|\\\\", " "),
    (r"\^\s*\{\s*\\prime\s*\}|\^\s*\\prime", "'"),
]
# equation numbers such as (2.1) are layout, not math; we strip them before speech anyway
EQ_NUM = re.compile(r"(?:\\mathrm\s*\{\s*\{?)?\(\s*\d+\s*\.\s*\d+\s*\)\}?\}?")
TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[^\s{}]")


def tokens(latex):
    s = EQ_NUM.sub("", latex)
    for pat, rep in SUBS:
        s = re.sub(pat, rep, s)
    # \text{ on } and \text{on} are the same word
    s = re.sub(r"\\text\s*\{([^}]*)\}", lambda m: r"\text{" + re.sub(r"[\s~]", "", m.group(1)) + "}", s)
    return TOKEN.findall(s)


def score(pred, ref):
    """Token-level similarity in [0, 1] (1 = identical after normalisation)."""
    return difflib.SequenceMatcher(None, tokens(pred), tokens(ref), autojunk=False).ratio()


def load_set(name):
    """(crop path, truth LaTeX) for every crop of crops/<name> that has ground truth.

    Uses refs/<name>_ground_truth.json (from build_truth.py) if present, else the p5-style .tex list."""
    gt = ROOT / "refs" / f"{name}_ground_truth.json"
    if gt.exists():
        return [(ROOT / "crops" / name / g["file"], g["truth"])
                for g in json.loads(gt.read_text(encoding="utf-8")) if g["truth"]]
    return list(zip(load_crops(name), load_truth(name)))


def load_crops(name="p5"):
    d = ROOT / "crops" / name
    return [d / m["file"] for m in json.loads((d / "manifest.json").read_text(encoding="utf-8"))]


def load_truth(name="p5"):
    lines = (ROOT / "refs" / f"{name}_ground_truth.tex").read_text(encoding="utf-8").splitlines()
    return [l for l in lines if l.strip() and not l.startswith("%")]
