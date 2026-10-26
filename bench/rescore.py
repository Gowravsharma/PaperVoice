"""
rescore.py - score saved recognizer outputs against the arXiv ground truth (no models run).

Reads out/bench/paddle.json and the CodeFormula LaTeX in out/profile/<config>.md.
"""
import json
import re

from common import ROOT, load_truth, score, tokens


def codeformula(config):
    md = (ROOT / "out" / "profile" / f"{config}.md").read_text(encoding="utf-8")
    return re.findall(r"^\$\$(.*)\$\$$", md, flags=re.M)


def report(name, preds, truth):
    s = [score(p, t) for p, t in zip(preds, truth)]
    print(f"{name:<24} " + "  ".join(f"{x:.2f}" for x in s) + f"   mean {sum(s) / len(s):.3f}")


def main():
    truth = load_truth()
    print(f"{'model':<24} " + "  ".join(f"#{i + 1:<3}" for i in range(len(truth))))
    for cfg in ("light", "light_egret"):
        report(f"CodeFormulaV2 ({cfg})", codeformula(cfg), truth)
    for name, res in json.loads((ROOT / "out" / "bench" / "paddle.json").read_text(encoding="utf-8")).items():
        report(name, [r["latex"] for r in res["rows"]], truth)
    if __debug__ and "-v" in __import__("sys").argv:
        for t in truth:
            print(" ".join(tokens(t)))


if __name__ == "__main__":
    main()
