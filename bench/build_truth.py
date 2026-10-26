r"""
build_truth.py - align Docling formula crops with display equations in the paper's arXiv LaTeX source.

Usage:
    python build_truth.py refs/arxiv_0905.1863/FahimTouziWarin-final.tex crops/fahim refs/fahim_ground_truth.json

Steps:
  1. expand the author's \def / \newcommand macros (whole control words only, so \i never eats \int)
  2. cut every display block (\b* \e*, \be \ee, equation, eqnarray, align, $$, \[ \]) into rows at top-level \\
  3. align crops (reading order) to rows (source order) with a monotone dynamic programme, comparing the
     crop's PDF text layer with a plain-text rendering of the LaTeX. One crop may cover several consecutive
     rows of one block; crops and rows may stay unmatched. No recognizer output is used, so the truth is
     independent of the models being scored.
"""
import difflib
import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------ LaTeX helpers
def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def group(s, i):
    """s[i] == '{' -> (content, index after the matching '}')."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "\\":
            continue
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return s[i + 1:], len(s)


def parse_macros(preamble):
    macros = {}
    for m in re.finditer(r"\\def\s*\\([a-zA-Z]+|.)\s*((?:#\d)*)\s*\{", preamble):
        body, _ = group(preamble, m.end() - 1)
        macros[m.group(1)] = (len(m.group(2)) // 2, body)
    for m in re.finditer(r"\\(?:re)?newcommand\s*\{?\s*\\([a-zA-Z]+)\s*\}?\s*(?:\[(\d)\])?\s*\{", preamble):
        body, _ = group(preamble, m.end() - 1)
        macros[m.group(1)] = (int(m.group(2) or 0), body)
    # environment shorthands are handled by the block finder, keep them out of expansion
    for k in ("be", "ee", "b", "e", "proof", "ep", "no", "point", "theequation"):
        macros.pop(k, None)
    return macros


def expand(s, macros, rounds=6):
    names = sorted(macros, key=len, reverse=True)
    words = [n for n in names if n[0].isalpha()]
    syms = [n for n in names if not n[0].isalpha()]
    pat = re.compile(r"\\(" + "|".join(map(re.escape, words)) + r")(?![a-zA-Z])"
                     + (r"|\\(" + "|".join(map(re.escape, syms)) + ")" if syms else ""))
    for _ in range(rounds):
        out, i, changed = [], 0, False
        for m in pat.finditer(s):
            if m.start() < i:
                continue
            name = m.group(1) or m.group(2)
            nargs, body = macros[name]
            j, args = m.end(), []
            for _ in range(nargs):
                while j < len(s) and s[j] == " ":
                    j += 1
                if j < len(s) and s[j] == "{":
                    a, j = group(s, j)
                else:
                    tok = re.match(r"\\[a-zA-Z]+|\\.|.", s[j:])
                    a, j = (tok.group(0), j + len(tok.group(0))) if tok else ("", j)
                args.append(a)
            for k, a in enumerate(args, 1):
                body = body.replace(f"#{k}", a)
            out.append(s[i:m.start()] + "{" + body + "}")
            i, changed = j, True
        s = "".join(out) + s[i:]
        if not changed:
            break
    return s


BLOCK = re.compile(
    r"\\b\*(.*?)\\e\*"
    r"|\\be(?![a-zA-Z])(.*?)\\ee(?![a-zA-Z])"
    r"|\\begin\{(equation|eqnarray|align|gather|multline)\*?\}(.*?)\\end\{\3\*?\}"
    r"|\$\$(.*?)\$\$"
    r"|\\\[(.*?)\\\]", re.S)


def split_rows(body):
    """Split at top-level \\ (not inside braces or nested environments)."""
    rows, depth, env, start, i = [], 0, 0, 0, 0
    while i < len(body):
        if body.startswith("\\begin", i):
            env += 1
        elif body.startswith("\\end", i):
            env -= 1
        c = body[i]
        if c == "\\" and i + 1 < len(body) and body[i + 1] == "\\":
            if depth == 0 and env == 0:
                rows.append(body[start:i])
                start = i + 2
            i += 2
            continue
        if c == "\\":
            i += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        i += 1
    rows.append(body[start:])
    return rows


def clean_row(r):
    r = re.sub(r"\\(?:label|tag)\s*\{[^}]*\}|\\nonumber|\\notag|\\nn\b", "", r)
    r = re.sub(r"\\(?:vspace|hspace)\*?\s*\{[^}]*\}|\\(?:no)?pagebreak|\\displaybreak", "", r)
    r = re.sub(r"\\lefteqn\s*", "", r)
    r = r.replace("&", " ")
    r = re.sub(r"\s+", " ", r).strip()
    return r


# ------------------------------------------------------------------ plain-text atoms for alignment
GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi "
         "rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega").split()
UNI = dict(zip("αβγδεεζηθϑικλμνξπρστυϕφχψωΓΔΘΛΞΠΣΦΨΩ", [g.lower() for g in
           "alpha beta gamma delta epsilon epsilon zeta eta theta theta iota kappa lambda mu nu xi pi rho sigma "
           "tau upsilon phi phi chi psi omega gamma delta theta lambda xi pi sigma phi psi omega".split()]))
UNI.update({"ε": "epsilon", "ϵ": "epsilon", "φ": "phi", "µ": "mu"})
ALIAS = {"varepsilon": "epsilon", "varphi": "phi", "vartheta": "theta"}


def atoms_latex(s):
    s = re.sub(r"\\(?:text|mbox|rm|mathrm|hbox)\s*\{([^}]*)\}", lambda m: " " + m.group(1) + " ", s)
    out = []
    for t in re.findall(r"\\[a-zA-Z]+|[A-Za-z0-9]|[=<>+\-(),\[\]|]", s):
        if t.startswith("\\"):
            name = t[1:]
            if name in GREEK:
                out.append(ALIAS.get(name, name).lower())
            elif name in ("sup", "inf", "max", "min", "lim", "log", "exp", "sin", "cos"):
                out.extend(name)
        else:
            out.append(t)
    return out


def atoms_pdf(s):
    s = re.sub(r"\(\s*\d+\s*\.\s*\d+\s*\)", " ", s)  # equation numbers
    out = []
    for ch in s:
        if ch in UNI:
            out.append(UNI[ch])
        elif ch.isascii() and (ch.isalnum() or ch in "=<>+-(),[]|"):
            out.append(ch)
        elif ch == "−":
            out.append("-")
    return out


def sim(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# ------------------------------------------------------------------ alignment
def align(crops, rows, max_span=12, band=120, gain=0.45):
    """Monotone DP. Matching crop i to rows [j, j+k) of one block scores sim - gain (only good matches pay)."""
    n, m = len(crops), len(rows)
    NEG = float("-inf")
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    best[0] = [0.0] * (m + 1)  # leading rows may be skipped freely
    for j in range(m + 1):
        back[0][j] = ("skip_row",) if j else None
    for i in range(1, n + 1):
        centre = i * m // n
        lo, hi = max(0, centre - band), min(m, centre + band)
        for j in range(m + 1):
            # crop i-1 unmatched
            if best[i - 1][j] > best[i][j]:
                best[i][j], back[i][j] = best[i - 1][j], ("skip_crop",)
            if j and best[i][j - 1] > best[i][j]:
                best[i][j], back[i][j] = best[i][j - 1], ("skip_row",)
            if not (lo <= j <= hi) or j == 0:
                continue
            # crop i-1 covers rows [j-k, j)
            joined = []
            for k in range(1, max_span + 1):
                s = j - k
                if s < 0 or rows[s]["block"] != rows[j - 1]["block"]:
                    break
                joined = rows[s]["atoms"] + joined
                v = best[i - 1][s]
                if v == NEG:
                    continue
                score = v + sim(crops[i - 1]["atoms"], joined) - gain
                if score > best[i][j]:
                    best[i][j], back[i][j] = score, ("match", k)
    # trace back
    i, j, match = n, m, {}
    while i > 0:
        step = back[i][j]
        if step[0] == "skip_crop":
            i -= 1
        elif step[0] == "skip_row":
            j -= 1
        else:
            k = step[1]
            match[i - 1] = list(range(j - k, j))
            i, j = i - 1, j - k
    return match


def source_rows(tex_path):
    """Every display-math row of the paper: {block, latex (macros expanded), atoms}."""
    tex = strip_comments(Path(tex_path).read_text(encoding="utf-8", errors="replace"))
    pre, body = tex.split("\\begin{document}", 1)
    macros = parse_macros(pre)
    rows = []
    for b, m in enumerate(BLOCK.finditer(body)):
        content = next(g for g in (m.group(1), m.group(2), m.group(4), m.group(5), m.group(6)) if g is not None)
        for r in split_rows(content):
            latex = clean_row(expand(r, macros))
            if latex:
                rows.append({"block": b, "latex": latex, "atoms": atoms_latex(latex)})
    return rows


def main():
    tex_path, crop_dir, out_path = map(Path, sys.argv[1:4])
    rows = source_rows(tex_path)
    b = rows[-1]["block"]

    manifest = json.loads((crop_dir / "manifest.json").read_text(encoding="utf-8"))
    crops = [{**c, "atoms": atoms_pdf(c["orig"])} for c in manifest]
    match = align(crops, rows)

    out = []
    for i, c in enumerate(crops):
        idx = match.get(i)
        truth = " \\\\ ".join(rows[r]["latex"] for r in idx) if idx else None
        joined = [a for r in idx for a in rows[r]["atoms"]] if idx else []
        out.append({"file": c["file"], "page": c["page"], "truth": truth,
                    "rows": len(idx) if idx else 0, "match": round(sim(c["atoms"], joined), 3),
                    "orig": c["orig"]})
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    good = sum(1 for o in out if o["truth"] and o["match"] >= 0.8)
    weak = sum(1 for o in out if o["truth"] and o["match"] < 0.8)
    print(f"{len(rows)} source rows in {b + 1} blocks; {len(crops)} crops: "
          f"{good} matched >=0.8, {weak} weak, {len(crops) - good - weak} unmatched -> {out_path}")


if __name__ == "__main__":
    main()
