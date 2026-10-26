"""
timing.py - when each word is spoken, and which printed word it belongs to (for word-by-word highlighting).

The voice model decides how many audio samples each phoneme gets (VITS duration predictor). With the voice
file patched to output those durations (read_aloud.Narrator loads <voice>.aligned.onnx, made once with Piper's
patch_voice_with_alignment), every word's start and end is exact: word_times_aligned() adds up the samples of
its phonemes; the spaces and the pauses at commas and full stops fall between words.

word_times() is the fallback for a voice without that output: each chunk's duration is shared out over its
phonemes by count, with extra weight for long vowels and pauses - an estimate that drifts where the voice
speaks some sounds faster or slower than average (the highlight then runs a word or two ahead or behind).

The spoken text differs from the print ("2019" -> "twenty nineteen", "σ" -> "sigma", citations dropped), so
spoken words are matched back to printed words: each printed word is turned into what it would sound like,
the two word lists are aligned, and unmatched spoken words take the printed word next to them.
"""
import difflib
import re

from pdf_reader.text_norm import normalize, prose, symbols

MARKS = set("ˈˌ")  # stress marks: not sounds
PAUSE = {",": 2.5, ";": 3.0, ":": 3.0, ".": 4.0, "!": 4.0, "?": 4.0}


def word_times(chunks, tokens):
    """chunks: [(phonemes list, seconds)] from the voice, tokens: the spoken words.
    Returns [(start, end)] per token, in seconds from the start of the audio."""
    spans, clock = [], 0.0
    for phonemes, seconds in chunks:
        words, cur, weights = [], 0.0, []
        for ph in phonemes:
            if ph == " ":
                words.append(cur)
                cur = 0.0
            elif ph in PAUSE:
                weights.append((len(words), PAUSE[ph]))  # pause after the current word
            elif ph == "ː":
                cur += 0.6
            elif ph not in MARKS:
                cur += 1.0
        words.append(cur)
        pause_after = {}
        for w, extra in weights:
            pause_after[min(w, len(words) - 1)] = pause_after.get(min(w, len(words) - 1), 0) + extra
        total = sum(words) + sum(pause_after.values()) or 1.0
        t = clock
        for i, units in enumerate(words):
            d = seconds * units / total
            spans.append((t, t + d))
            t += d + seconds * pause_after.get(i, 0) / total
        clock += seconds
    if not tokens:
        return []
    if len(spans) == len(tokens):
        return spans
    # the voice split words differently (e.g. "one-to-many"): map by position
    return [spans[min(len(spans) - 1, int(i * len(spans) / len(tokens)))] for i in range(len(tokens))]


BOUNDARY = {"^", "$", "_", " "}  # sentence start/end, padding, word gap (Piper's phoneme map)


def word_times_aligned(chunks, tokens, rate):
    """chunks: [(phoneme_alignments, number of samples)] - each alignment a (phoneme, num_samples) pair from the
    voice, in order. Returns [(start, end)] per token in seconds, or None if the words do not line up."""
    spans, sizes, offset = [], [], 0
    for alignments, n_samples in chunks:
        pos, start, size = offset, None, 0
        for ph, n in alignments:
            if ph in BOUNDARY or ph in PAUSE:
                if start is not None:
                    spans.append((start / rate, pos / rate))
                    sizes.append(size)
                    start = None
            else:
                if start is None:
                    start, size = pos, 0
                size += ph not in MARKS
            pos += n
        if start is not None:
            spans.append((start / rate, pos / rate))
            sizes.append(size)
        offset += n_samples
    if not spans or not tokens:
        return None
    if len(spans) == len(tokens):
        return spans
    # the voice joined or split words ("for a" is one word to it, "one-to-many" three): give each written word the
    # spoken word at the same relative place - letters for the written words, phonemes for the spoken ones - so
    # a join only affects its neighbours, not the rest of the sentence
    letters = [max(1, len(_key(t))) for t in tokens]
    total_l, total_s = sum(letters), sum(sizes) or len(sizes)
    bounds, acc = [], 0
    for s in sizes:
        acc += (s or 1) / total_s
        bounds.append(acc)
    out, acc, j = [], 0, 0
    for n in letters:
        mid = (acc + n / 2) / total_l
        acc += n
        while j < len(bounds) - 1 and bounds[j] < mid:
            j += 1
        out.append(spans[j])
    return out


def _key(word):
    return re.sub(r"[^\w]", "", word.lower())


def _sounds_like(word):
    """What a printed word turns into when read (same rules as the script), as comparable keys."""
    cite = re.fullmatch(r"\[(\d+)\][.,;:]?", word)
    if cite:  # alone, "[12]" would be dropped as a citation; in running text it is read "reference twelve"
        return ["reference"] + [k for k in map(_key, normalize(cite.group(1)).split()) if k]
    if re.search(r"[\d\W]", word, re.UNICODE) and not re.fullmatch(r"[\w'-]+[.,;:!?]?", word):
        spoken = normalize(symbols(prose(word)))
    elif re.search(r"\d", word):
        spoken = normalize(word)
    else:
        spoken = word
    return [k for k in map(_key, spoken.split()) if k]


def spoken_to_printed(printed, tokens):
    """For each spoken token, the index of the printed word (in printed.split()) it belongs to, or None."""
    words = printed.split()
    if not words:
        return [None] * len(tokens)
    expected, owner = [], []
    for i, w in enumerate(words):
        for k in _sounds_like(w):
            expected.append(k)
            owner.append(i)
    got = [_key(t) for t in tokens]
    out = [None] * len(tokens)
    sm = difflib.SequenceMatcher(None, expected, got, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                out[j1 + k] = owner[i1 + k]
        elif tag == "replace" and i2 > i1:
            for k in range(j2 - j1):  # spread the spoken words over the printed words they replace
                out[j1 + k] = owner[i1 + min(i2 - i1 - 1, k * (i2 - i1) // max(1, j2 - j1))]
    last = None
    for j in range(len(out)):  # spoken words with no printed counterpart stay on the previous word
        if out[j] is None:
            out[j] = last
        last = out[j]
    return out
