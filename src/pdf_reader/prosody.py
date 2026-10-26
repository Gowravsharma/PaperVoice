"""
prosody.py - where a spoken sentence should breathe, and how fast it should go (works on any text).

1. Prosodic phrasing of long sentences. People do not read a long sentence as one run, nor do they pause only at
   commas: speech is grouped into prosodic phrases, and the breaks fall at syntactic boundaries, balanced in
   length (Gee & Grosjean 1983, "Performance structures"; Bachenko & Fitzpatrick 1990, "A computational grammar
   of discourse-neutral prosodic phrasing in English"). A light-weight rule used by classic TTS systems is
   "chinks 'n chunks" (Liberman & Church 1992; Taylor & Black 1998, Festival): a phrase is a run of function words
   (chinks: the, of, in, which, ...) followed by content words (chunks), and a break may go BEFORE a function word
   that follows a content word. Strongest before conjunctions and relative/subordinating words (and, but, which,
   where, because, ...), then before prepositions; a break is placed when the current phrase is long enough and
   enough of the sentence is left (balanced phrases), never inside a run of math tokens. A break is written as
   a comma: the voice pauses ~0.16 s with a "more to come" melody.

2. Math inside sentences ("for a smooth function phi on QT"). The voice speaks symbol names as quickly as short
   words. Measured on this voice: it cannot be slowed for a single word (lengthening or doubling vowels in the
   phonemes changed nothing; only its global pace works). So (a) a short pause goes before a run of math tokens,
   the way people read math aloud ("a smooth function, phi, on QT"), and after it when the sentence goes on with
   ordinary words; (b) the sentence is read slower in proportion to how much of it is math (speed_factor).

Math tokens are recognised by form only: Greek letter names, single letters (not the article "a" or "I"; the
variable a is already "eigh", see text_norm.inline_math), short symbol tokens such as "QT", "Rd", "R+", "Lp", and the
words that join them inside a run ("plus", "equals", "is defined as", "sub", "of", "comma", ...).
"""
import re

GREEK = set("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda lamda mu nu xi omicron pi rho sigma "
            "tau upsilon phi chi psi omega".split())
# words that join math tokens inside one run: "phi of t comma x", "R plus times Rd", "a is defined as b"
MATH_JOIN = set("plus minus times equals is defined as sub superscript of comma open close paren brace bracket in "
                "less greater than or equal to over squared cubed inverse prime hat bar tilde dot transpose not "
                "wedge vee cap cup circ star intersection union min max sup inf lim log exp sin cos dot "
                "capital zero one two three four five six seven eight nine ten infinity".split())
CLOSERS = {"paren", "brace", "bracket"}  # a run may end with "... close paren"
OPENERS = {"open", "paren", "brace", "bracket", "capital", "negative", "minus"}  # ... and start with "open paren"
# words that lead straight into a symbol, with no pause: "let mu", "denote by X", "where f is"
LEAD_INS = set("let denote denotes denoted define defines defined call called set fix take put consider "
               "write writes written assume suppose given using see".split())
COMMON_SHORT = set("of to in on at by is as an be or if it we so no up do go he me my us am ok".split())

# strong breaks: words that open a clause. "and"/"or" often join two nouns ("existence and uniqueness"), so
# they only count as weak breaks; "of" never starts a phrase of its own ("a justification of our scheme")
CONJUNCTIONS = set("but which where when while whereas because since although though unless whether hence thus "
                   "therefore whose whom who".split())
WEAK = {"and", "or", "nor", "that", "if", "then", "so", "yet"}
PREPOSITIONS = set("in on at by for with without within from into onto over under between among through during "
                   "after before against along across about around toward towards upon via per".split())
FUNCTION = CONJUNCTIONS | WEAK | PREPOSITIONS | set(
    "the a an this that these those its their our his her my your of to as is are was were be been being has have "
    "had do does did can could may might must shall should will would it we they he she i you which there "
    "not no all each every some any".split())


def is_math_token(w):
    k = re.sub(r"[^\w+]", "", w)
    if not k:
        return False
    low = k.lower()
    if low in GREEK or low == "eigh":
        return True
    if len(k) == 1 and k.isalpha():
        return k not in ("a", "I", "A")
    if low in COMMON_SHORT:
        return False
    # short symbol tokens: Rd, Ft, Lp (flattened sub/superscripts), R+; not all-caps words (TTS, LN: acronyms)
    return bool(re.fullmatch(r"[A-Z][a-z]\+?|[a-z][A-Z]|[A-Z]\+", k))


def math_runs(words):
    """[start, end) word spans that are math: math tokens, plus joining words between two math tokens."""
    runs, i, n = [], 0, len(words)
    while i < n:
        if not is_math_token(words[i]):
            i += 1
            continue
        j = i + 1
        while j < n:
            if is_math_token(words[j]):
                j += 1
                continue
            k = j  # joining words, if another math token follows them
            while k < n and re.sub(r"[^\w]", "", words[k]).lower() in MATH_JOIN and not is_math_token(words[k]):
                k += 1
            if k < n and k > j and is_math_token(words[k]) and k - j <= 7:
                j = k + 1
                continue
            if k > j and _bare(words[k - 1]) in CLOSERS:  # "open paren zero comma T close paren"
                j = k
                continue
            break
        a = i
        while a > 0 and _bare(words[a - 1]) in OPENERS and not (runs and a - 1 < runs[-1][1]):
            a -= 1
        runs.append((a, j))
        i = j
    return runs


def _bare(w):
    return re.sub(r"[^\w]", "", w).lower()


def phrase(text, min_words=5, long_sentence=12):
    """Add pauses (commas) to one spoken sentence: before runs of math tokens, and prosodic phrase breaks in
    long sentences. Returns (text, speed_factor). Words are never changed."""
    words = text.split()
    n = len(words)
    if n == 0:
        return text, 1.0
    runs = math_runs(words)
    in_math = set(k for a, b in runs for k in range(a, b))
    pause_after = set()  # indices of words after which a comma goes

    # (a) pauses around math runs inside a sentence: before the symbol when it follows a noun it names
    # ("a smooth function, phi"; not after "let", "and", "from": those already lead into it), and after a run
    # of three or more words when the sentence goes on ("R plus times Rd to Rd, and M of d comma d")
    for a, b in runs:
        prev = words[a - 1] if a > 0 else ""
        between_symbols = a >= 2 and is_math_token(words[a - 2])  # "a wedge b", "A times B": one expression
        if (prev and not between_symbols and not re.search(r"[,;:]$", prev)
                and _bare(prev) not in FUNCTION | LEAD_INS | MATH_JOIN):
            pause_after.add(a - 1)
        if b < n and n - b >= 3 and b - a >= 3 and not re.search(r"[,;:.]$", words[b - 1]):
            pause_after.add(b - 1)

    # (b) prosodic phrase breaks (chinks 'n chunks with balancing) in long sentences
    if n >= long_sentence:
        run_len = 0
        for i in range(n):
            if i in pause_after or re.search(r"[,;:]$", words[i]):
                run_len = 0
                continue
            run_len += 1
            nxt = words[i + 1] if i + 1 < n else ""
            left = n - (i + 1)
            if not nxt or i in in_math or (i + 1) in in_math or left < 5:
                continue
            prev_content = _bare(words[i]) not in FUNCTION and re.search(r"[A-Za-z]", words[i])
            nb = _bare(nxt)
            if not prev_content or nb not in CONJUNCTIONS | WEAK | PREPOSITIONS or nb == "of" or nxt[0] in "([":
                continue
            strength = 3 if nb in CONJUNCTIONS else 2 if nb in PREPOSITIONS else 1
            need = {3: min_words, 2: min_words + 5, 1: min_words + 7}[strength]
            if run_len >= need:
                pause_after.add(i)
                run_len = 0

    out = [w + "," if i in pause_after and not re.search(r"[,;:.!?]$", w) else w for i, w in enumerate(words)]
    math_share = len(in_math) / n
    speed_factor = 1.0 - 0.25 * min(1.0, math_share * 2)  # up to 25 % slower for math-dense sentences
    return " ".join(out), round(speed_factor, 3)
