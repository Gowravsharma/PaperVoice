r"""
latex_cleanup.py - make recognizer LaTeX read well aloud.

tidy(latex) -> (latex, equation_number or None)

  1. equation numbers such as "(2.4)\quad ..." or "... \qquad (2.3)" are removed and returned separately
     (also garbled forms the model produces: "\ 3.29)", "\big(2.2\big)", "\mathrm{(2.1)}").
  2. letter-spaced groups are re-joined: "\mathrm{a n d}" -> "and"; inside a group "~" separates words:
     "\mathrm{i s~a~s u b}" -> "is a sub".
  3. known operators become commands: "\operatorname*{lim sup}" -> "\limsup", "\mathrm{exp}" -> "\exp".
  4. remaining multi-letter words in \mathrm / \mathit become "\text{ word }", so speech engines read a word
     instead of spelling letters; unknown \operatorname names are only re-joined. Single letters
     (e.g. ^\mathrm{T}) are kept.
  5. bare letter-spaced groups whose words are all in WORDS ("{a n d}", "{l o c a l l y~u n i f o r m l y}",
     where the model dropped \mathrm) become text too. Other bare groups may be products ({a b c}) and stay.
"""
import re

OPERATORS = {"sup", "inf", "lim", "limsup", "liminf", "max", "min", "exp", "ln", "log", "det", "dim",
             "ker", "sin", "cos", "tan", "arg", "deg", "gcd", "Pr"}
WORDS = {"and", "or", "on", "for", "all", "any", "some", "each", "every", "where", "with", "if", "then",
         "is", "of", "in", "as", "resp", "locally", "uniformly", "otherwise", "else", "such", "that"}
GROUP = re.compile(r"\\(operatorname\*?|mathrm|mathit|text|textrm)\s*\{([^{}]*)\}")
BARE = re.compile(r"(?<![\^_])\{((?:[a-z](?: |~))+[a-z]~?)\}")
SPACE = r"(?:\\quad|\\qquad|\\[,;:! ]|~|\s)*"
NUM = r"(?:\\mathrm\s*\{\s*\{?\s*)?(?:\(|\\big\(|\\left\(|\\\s)\s*(\d+\s*\.\s*\d+)\s*(?:\)|\\big\)|\\right\))\s*\}?\s*\}?"
LEAD = re.compile(r"^(\s*(?:\\begin\{aligned\}\s*\{?\s*\{?\s*)?)" + NUM + SPACE)
TRAIL = re.compile(SPACE + NUM + r"\s*$")
# plain integer numbers "(1)" only when clearly set apart (\quad, \eqno, \tag), so f(1) is never touched
TRAIL_INT = re.compile(r"(?:\\quad|\\qquad|\\eqno|\\tag)\s*\{?\s*\(\s*(\d+)\s*\)\s*\}?\s*$")


def _join(content):
    """'a n d' -> 'and'; 'i s~a~s u b' -> 'is a sub'; leaves normal text alone."""
    words = [w for w in re.split(r"~|\\ ", content)]
    out = []
    for w in words:
        parts = w.split()
        if len(parts) > 1 and all(len(p) == 1 or p.endswith(".") and len(p) == 2 for p in parts):
            out.append("".join(parts))
        else:
            out.append(" ".join(parts))
    return " ".join(p for p in out if p)


def _group(m):
    cmd, content = m.group(1), m.group(2)
    if not re.fullmatch(r"[A-Za-z .~\\]*", content):
        return m.group(0)
    word = _join(content)
    if word.replace(" ", "") in OPERATORS and (cmd.startswith("operatorname") or cmd == "mathrm"):
        return "\\" + word.replace(" ", "") + " "
    if cmd in ("text", "textrm"):
        return "\\text{" + word + "}"
    if len(word.replace(" ", "")) <= 1:
        return m.group(0)
    if cmd.startswith("operatorname"):
        return "\\" + cmd + "{" + word + "}"
    return "\\text{ " + word + " }"


def _bare(m):
    word = _join(m.group(1))
    return "\\text{ " + word + " }" if all(w in WORDS for w in word.split()) else m.group(0)


def tidy(latex):
    number = None
    m = LEAD.match(latex)
    if m:
        number = re.sub(r"\s", "", m.group(2))
        latex = m.group(1) + latex[m.end():]
    else:
        m = TRAIL.search(latex) or TRAIL_INT.search(latex)
        if m:
            number = re.sub(r"\s", "", m.group(1))
            latex = latex[:m.start()]
    latex = GROUP.sub(_group, latex)
    latex = BARE.sub(_bare, latex)
    latex = re.sub(r"\{\s*(\\text\{[^{}]*\})\s*\}", r"\1", latex)  # "{\text{ and }}" -> "\text{ and }"
    return latex.strip(), number
