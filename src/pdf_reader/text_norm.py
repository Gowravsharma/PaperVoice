"""
text_norm.py - make script text speakable. Three layers, applied in this order:

1. prose(text)       document structure, body text only (decided with the user, 2026-10-22):
                     citations "[12]", "(Ren et al., 2019)" dropped; "(2.1)" -> "equation 2.1";
                     URLs -> "a link", e-mail addresses -> "an e-mail address", arXiv ids; fixed abbreviations
                     (e.g., i.e., et al., ...); remaining square brackets -> parentheses (wetext cannot parse "[").
2. symbols(text)     literal symbol words (v1: no meaning from context): ±, ×, %, relations, Greek letters by
                     name, and symbols left by inline math flattened in the PDF text (=, +, √, ∑, ...);
                     pure markers (†, ✷, ‖, accents) are dropped. Shared with tables.
3. normalize(text)   semiotic classes - numbers, decimals, years, dates, ordinals, money, phone numbers - by a
                     real WFST text normaliser (wetext: WeNet's pre-compiled English grammars on kaldifst;
                     NeMo/pynini does not build on Windows). Only sentences with a digit or symbol go through it;
                     a sentence it cannot parse is kept as it is and counted in FAILURES.
"""
import re
import unicodedata

GREEK_FIX = {"∆": "delta", "µ": "mu", "ϵ": "epsilon", "ϕ": "phi"}
RELATIONS = {"≪": "much less than", "≫": "much greater than", "≤": "less than or equal to",
             "≥": "greater than or equal to", "≈": "approximately", "≠": "not equal to",
             "∈": "in", "→": "to", "∞": "infinity", "√": "square root of", "∑": "sum", "∏": "product",
             "∫": "integral", "∇": "nabla", "∂": "partial", "≡": "equivalent to", "′": "prime", "·": "times",
             "=": "equals", "&": "and", "<": "less than", ">": "greater than", "∧": "wedge", "∨": "vee",
             "⊕": "circled plus", "⊗": "circled times",
             # standard spoken names where the Unicode name is odd ("white bullet", "up tack")
             "◦": "circle", "∘": "circle", "↦": "maps to", "⊥": "perpendicular to", "↪": "hooked arrow",
             "⇢": "dashed arrow", "∼": "tilde", "≲": "less than or similar to", "≍": "asymptotic to"}
DROP = set("†‡§✷∗*‖∥|¯˜¨´`^̸□■∎◻")  # □ ∎: end-of-proof marks  # footnote markers, norm bars and accents split off by the PDF text layer

ABBREVIATIONS = {"e.g.": "for example", "i.e.": "that is", "et al.": "and others", "resp.": "respectively",
                 "cf.": "compare", "vs.": "versus", "w.r.t.": "with respect to", "a.s.": "almost surely",
                 "r.v.": "random variable", "i.i.d.": "independent and identically distributed"}
CITE_NUM = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
CITE_AUTHOR = re.compile(r"\s*\((?:see\s+|e\.g\.,?\s+)?(?:[A-Z][^()]*?,?\s*(?:19|20)\d{2}[a-z]?(?:;\s*|,\s*)?)+\)")
CITE_YEAR = re.compile(r"(?<=et al\.)\s*\((?:19|20)\d{2}[a-z]?\)|(?<=[a-z])\s+\((?:19|20)\d{2}[a-z]?\)")
EQ_REF = re.compile(r"\((\d+\.\d+)\)")
URL = re.compile(r"https?://\S+?(?=[.,;)]?(?:\s|$))|www\.\S+?(?=[.,;)]?(?:\s|$))")
EMAIL = re.compile(r"\{?[\w.,-]+\}?@[\w-]+(?:\.[\w-]+)+")


def greek(ch):
    if ch in GREEK_FIX:
        return GREEK_FIX[ch]
    # lookalikes with their own code point: Ω OHM SIGN -> Ω, µ MICRO SIGN -> μ (fonts use them for the letters)
    ch = unicodedata.normalize("NFKC", ch)
    if len(ch) != 1:
        return None
    name = unicodedata.name(ch, "")
    if name.startswith("GREEK SMALL LETTER "):
        return name.removeprefix("GREEK SMALL LETTER ").lower()
    if name.startswith("GREEK CAPITAL LETTER "):
        return "capital " + name.removeprefix("GREEK CAPITAL LETTER ").lower()
    if name.startswith("GREEK ") and name.endswith(" SYMBOL"):  # ϑ "GREEK THETA SYMBOL", ϖ, ϱ, ϰ
        return name.removeprefix("GREEK ").removesuffix(" SYMBOL").lower()
    return None


def unicode_word(ch):
    """Fallback for any other math/other symbol: its official Unicode name, literally ("⊂" -> "subset of",
    "∅" -> "empty set"). Combining accents (the hat in "f̂", split off by the text layer) are dropped."""
    cat = unicodedata.category(ch)
    if cat == "Mn":
        return ""
    if cat in ("Sm", "So"):
        return unicodedata.name(ch, "").lower() or None
    return None


def _spaced(t, i, word):
    """word for t[i], with a space only where it touches a letter or digit ("µ-law" -> "mu-law")."""
    before = " " if i > 0 and t[i - 1].isalnum() else ""
    after = " " if i + 1 < len(t) and t[i + 1].isalnum() else ""
    return before + word + after


ARXIV = re.compile(r"arXiv:\s*\S+")


def prose(t):
    t = EMAIL.sub("an e-mail address", t)
    t = URL.sub("a link", t)
    t = ARXIV.sub("an arXiv identifier", t)  # "math.PR/0905.1863" is an identifier, not a number
    # "suggested in [12]": the citation is part of the sentence -> "in reference 12"; elsewhere it is dropped
    t = re.sub(r"\b(in|of|see|from|by|to|and|also|cf\.)\s*\[(\d+(?:\s*,\s*\d+)*)\]",
               lambda m: f"{m.group(1)} reference{'s' * (',' in m.group(2))} "
                         + " and ".join(x.strip() for x in m.group(2).split(",")), t)
    t = CITE_NUM.sub("", t)
    t = CITE_AUTHOR.sub("", t)
    t = CITE_YEAR.sub("", t)
    t = re.sub(r"\s+\b(and|or)\s*(?=[,.;])", "", t)  # "in [12] and (Ren et al., 2019)," left "… and,"
    t = EQ_REF.sub(r"equation \1", t)
    for short, long in ABBREVIATIONS.items():
        t = re.sub(r"(?<![\w.])" + re.escape(short) + r"(?=[\s,;:.)]|$)", long, t)  # "et al.." too
    t = t.replace("[", "(").replace("]", ")")
    return re.sub(r"\s+([,.;:])", r"\1", t)


MATHY_ITEM = re.compile(r"[\wͰ-Ͽ′'+\-·.]{1,4}")
BRACKET_WORDS = {"(": "open paren", ")": "close paren", "{": "open brace", "}": "close brace"}


# the letter a as a variable: the voice reads "a" as the article "uh" and "ay" as "eye"; "eigh" is the letter name
LETTER_A = "eigh"
# two-letter English words; any other two lower-case letters next to an operator are flattened math ("ti = ih" is
# t_i = i·h), spelled out letter by letter, or the voice reads "ti" as "tie" and "ih" as "eye"
TWO_LETTER_WORDS = set("am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we ie eg vs "
                       "ex ad id ah ax".split())


def _item(x):
    x = x.strip()
    return LETTER_A if x == "a" else x  # inside a list "a" is the variable, never the article


def _mathy(content):
    """A comma-separated list of short items (symbols, letters, numbers): "d, d", "0, T", "Ω, F, F, P"."""
    items = [x.strip() for x in content.split(",")]
    return len(items) >= 2 and all(MATHY_ITEM.fullmatch(x) for x in items)


def inline_math(t):
    """Math written inline in a sentence arrives as flattened text, and the voice silently skips its brackets
    and commas ("M(d, d)" was heard as "M d d"). Give them a spoken form, in the same style as display formulas:
      f(x, y)      -> "f of x comma y"            (a short name right before a bracketed list)
      (0, T), {..} -> "open paren 0 comma T close paren"
      a := ..., x = a + 1 -> "eigh" for the variable a (the voice reads a lone "a" as the article "uh")
      ti = ih      -> "t i = i h" (two letters that are not a word, next to an operator: flattened math)
    Only short-item lists count, so ordinary parentheses in prose ("see Section 2", "i.e., ...") are untouched."""
    def call(m):
        if not _mathy(m.group(2)):
            return m.group(0)
        return f"{m.group(1)} of {' comma '.join(_item(x) for x in m.group(2).split(','))}"
    t = re.sub(r"(?<![\wͰ-Ͽ])([A-Za-zͰ-Ͽ][\wͰ-Ͽ]{0,2})\(([^(){}]{1,40})\)", call, t)

    def group(m):
        if not _mathy(m.group(2)):
            return m.group(0)
        items = " comma ".join(_item(x) for x in m.group(2).split(","))
        return f" {BRACKET_WORDS[m.group(1)]} {items} {BRACKET_WORDS[m.group(3)]}"  # "max{a, 0}": a space
    t = re.sub(r"([({])([^(){}]{1,40})([)}])", group, t)
    ops = r"(?::=|=|≤|≥|<|>|\+|−|∈|×|·|∧|∨|∩|∪)"
    t = re.sub(r"(?<![\w'’])a(?=\s*" + ops + ")", LETTER_A, t)
    t = re.sub(r"(" + ops + r")(\s*)a(?![\w'’])", r"\1\2" + LETTER_A, t)

    # "i = 0, . . . , n": the voice swallows the dots ("zero-en"); said literally, as in display formulas
    # only between commas (a list of values); "..." in prose is left to the voice
    t = re.sub(r"(?<=,)\s*(?:\.\s*){3}(?=,)|(?<=,)\s*…\s*(?=,)", " dot dot dot", t)

    def spell(m):
        w = m.group(0)
        return w if w in TWO_LETTER_WORDS else " ".join(w)
    t = re.sub(r"(?<![\w'’])[a-z]{2}(?=\s*" + ops + r")", spell, t)
    t = re.sub(r"(?<=" + "|".join(re.escape(o) for o in "=≤≥<>+−∈×·∧∨∩∪") + r")(\s*)([a-z]{2})(?![\w'’])",
               lambda m: m.group(1) + spell(re.match(r".*", m.group(2))), t)
    return t


def symbols(t, relations=RELATIONS):
    t = inline_math(t)
    # a version letter after a CamelCase name is not a unit: "FastSpeech 2s" -> "2 S" (the voice says "ess"),
    # not "two seconds" (the number normaliser reads "2s" and "2 s" as seconds, but not "2 S")
    # "a/b" with a decimal or a version letter is not a fraction: "4.5/5" (was "four.five fifths"), "2/2s" (was
    # "two halves seconds") -> "slash"; plain "2/3" stays a fraction for the number normaliser ("two thirds")
    t = re.sub(r"(\d+\.\d+)\s*/\s*(?=\d)|(?<=\d)\s*/\s*(?=\d+\.\d|\d+[a-z]\b)", lambda m: (m.group(1) or "") + " slash ", t)
    t = re.sub(r"(\b[A-Z][a-z]+[A-Z]\w*\s+\d+(?: slash \d+)?)([a-z])\b", lambda m: f"{m.group(1)} {m.group(2).upper()}", t)
    t = re.sub(r"(\d)\s?[x×](?=[\s,.;:)]|$)", r"\1 times", t)  # "a 3x speed-up"
    t = re.sub(r"(\d)\s+/\s+(\d)", r"\1 slash \2", t)  # "6,139 / 255.8": two values, not a fraction
    t = re.sub(r"\s\+\s|(?<=[\w)])\+(?=[\w(])", " plus ", t)
    t = t.replace("±", " plus or minus ").replace("×", " times ").replace("%", " percent").replace("−", "-")
    # ":=" is read "is defined as", as in display formulas (speech/phrasing.mjs); "̸=" is ≠ split in two
    t = t.replace(":=", " is defined as ").replace("̸=", "≠")
    t = re.sub(r"(?<=\s)\\(?=\s)", "set minus", t)  # "X \ I(f)": the text layer stores ∖ as a backslash
    t = re.sub(r"[\"“”„]", "", t)  # quotation marks are not spoken, and wetext escapes them as \"
    t = re.sub(r"(?<=[(,])\s*·\s*(?=[,)])", "dot", t)  # placeholder: φ(t,·) -> "phi(t,dot)", not "times"
    out = []
    for i, ch in enumerate(t):
        if ch in DROP:
            out.append(" ")
            continue
        word = relations.get(ch) or greek(ch)
        if word is None and not ch.isascii():
            word = unicode_word(ch)
        out.append(_spaced(t, i, word) if word else (" " if word == "" else ch))
    t = re.sub(r"[ \t]+", " ", "".join(out))
    t = re.sub(r"\s+([,.;:])", r"\1", t)  # dropped markers leave "PDES ,."
    return re.sub(r"[,;:]+\.", ".", t).strip()


_normalizer = None
FAILURES = []  # sentences wetext could not parse (kept unnormalised)
NEEDS_TN = re.compile(r"[\d$€£%@#&/]")


def _get_normalizer():
    global _normalizer
    if _normalizer is None:
        from wetext import Normalizer
        _normalizer = Normalizer(lang="en", operator="tn")
    return _normalizer


_number_cache = {}


def normalize_numbers(t):
    """For spoken formulas: their only semiotic class is plain numbers ("0", "2.5"), so only those tokens go
    through wetext (cached). Whole formula sentences are long (up to ~370 words) and wetext's time grows
    with sentence length."""
    def word(m):
        tok = m.group(0)
        if tok not in _number_cache:
            try:
                _number_cache[tok] = _get_normalizer().normalize(tok)
            except Exception:
                FAILURES.append(tok)
                _number_cache[tok] = tok
        return _number_cache[tok]
    return re.sub(r"\d+(?:\.\d+)?", word, t)


WINDOW = 3  # words of context on each side of a token that needs normalising
# what wetext rewrites: digits, and any symbol besides letters and ordinary punctuation (+, _, \, #, *, ...)
WINDOW_TRIGGER = re.compile(r"[\d]|[^\w\s'’.,;:!?()\-]|_")
LONG = 14  # sentences longer than this are normalised in windows


def _normalize_sentence(s):
    """wetext's time grows with the length of what it gets (~1.8 ms per word), and most of a long sentence has
    nothing to normalise: only windows of WINDOW words around each token with a digit or symbol go through it
    (overlapping windows merged), enough context for "June 12, 2019" or "from 1990 to 2000". Checked on every
    sentence of the indexed papers: identical output (bench/check_normalize_windows.py)."""
    words = s.split(" ")
    if len(words) <= LONG:
        return _normalizer.normalize(s)
    spans = []
    for i, w in enumerate(words):
        if WINDOW_TRIGGER.search(w):
            a, b = max(0, i - WINDOW), min(len(words), i + WINDOW + 1)
            if spans and a <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], b)
            else:
                spans.append([a, b])
    out, pos = [], 0
    for a, b in spans:
        out += words[pos:a]
        out.append(_normalizer.normalize(" ".join(words[a:b])))
        pos = b
    out += words[pos:]
    return " ".join(x for x in out if x)


def normalize(t):
    """Semiotic classes via wetext, sentence by sentence; sentences without digits/symbols are skipped."""
    if not NEEDS_TN.search(t):
        return t
    _get_normalizer()
    out = []
    for s in re.split(r"(?<=[.!?;])\s+", t):
        if not NEEDS_TN.search(s):
            out.append(s)
            continue
        try:
            out.append(_normalize_sentence(s))
        except Exception:
            FAILURES.append(s)
            out.append(s)
    return " ".join(out)
