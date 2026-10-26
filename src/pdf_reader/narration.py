"""
narration.py - turn a page's index (blocks and sentences) into speech units with explicit pauses.

The voice (Piper) is given one unit at a time - a sentence, a formula, a heading, a table sentence - and the
planned silence is added after it. Measured on this voice: Piper itself leaves ~0.18 s around each sentence
it produces, and pauses ~0.16 s at a comma and ~0.30 s at a semicolon inside a sentence. Before this planner
the sentences of a paragraph were sent as one text, so they ran into each other with ~0.18 s between them.

Pause classes (seconds of silence ADDED after the unit, at speed 1.0; divided by the speed setting):
  clause     0.15  (kept for other short joins)
  formula    0.30  a lead-in ending in ":" before a display formula ("we define:"), or a formula followed by
                   a continuation that starts in lower case ("... of the diffusion X"): a listener needs a
                   moment to switch between words and math
  sentence   0.35  between sentences of a paragraph
  paragraph  0.70  after the last sentence of a paragraph, a table, a formula that ends a statement
  section    1.00  after a heading
These are starting values, to be tuned by listening.

Pace (relative to the user's speed; prosody.py): formulas are read at FORMULA_PACE, and a sentence with math
inside ("a smooth function phi on QT") is slowed in proportion to its share of math words. Long sentences get
prosodic phrase breaks, and symbols a pause before them (prosody.phrase); only commas are added, never words.

Each unit keeps its source (block id, sentence id, page, printed text) so it can be traced and highlighted.
Sentences are never split or reworded here: the narration planner only decides boundaries and pauses, so it
cannot change what a sentence or formula says (the math adapter in speech/phrasing.mjs decides the wording).
"""
from dataclasses import dataclass

import re

from pdf_reader.prosody import phrase
from pdf_reader.text_norm import LETTER_A

PAUSES = {"none": 0.0, "clause": 0.15, "formula": 0.30, "sentence": 0.35, "paragraph": 0.70, "section": 1.00}
FORMULA_PACE = 0.8  # display math 20 % slower than the text around it


@dataclass
class Unit:
    text: str  # what the voice says
    pause: str  # pause class after this unit (PAUSES)
    kind: str  # heading | text | caption | formula | table | announcement
    page: int
    block: str = ""
    sentence: str = ""
    source: str = ""  # the printed text ("" for formulas and tables)
    has_math: bool = False
    pace: float = 1.0  # multiplies the user's speed for this unit


def formula_voice(t):
    """Two things the voice gets wrong in a formula reading: the variable a is said as the article "uh" (the
    engine writes it "a"; its own "is a member of" is the article), and a pause the phrasing puts before a
    written comma is followed by the word "comma" ("equals zero; comma on [0, T)")."""
    t = re.sub(r"(?<!\bis )(?<!\bnot )\ba\b(?! (?:member|subset|proper|fraction|function)\b)", LETTER_A, t)
    return re.sub(r"([;,]) comma\b,?", r"\1", t)


def plan_page(blocks, page, tail=None, announce=False):
    """Speech units for one page. blocks: index.Block list in reading order. tail: the page-break sentence
    to say at the end (if the next page did not take it over). announce: start with "Page N."."""
    from pdf_reader.text_norm import normalize
    units = []
    if announce:
        units.append(Unit(normalize(f"Page {page}."), "sentence", "announcement", page))
    for bi, b in enumerate(blocks):
        nxt = blocks[bi + 1] if bi + 1 < len(blocks) else None
        for si, s in enumerate(b.sentences):
            last = si == len(b.sentences) - 1
            if b.kind == "heading":
                pause = "section"
            elif not last:
                pause = "sentence"
            elif b.kind == "formula":
                # a formula usually belongs to the sentence around it: if the text after it continues that
                # sentence ("of the diffusion X ..."), keep the pause short
                cont = nxt is not None and nxt.kind == "text" and nxt.sentences[0].text[:1].islower()
                pause = "formula" if cont else "paragraph"
            elif nxt is not None and nxt.kind == "formula" and (
                    s.text.rstrip().endswith((":", ",")) or s.text.rstrip()[-1:].isalpha()):
                pause = "formula"  # "we define:" / "Given a map" -> the formula follows directly
            else:
                pause = "paragraph"
            if b.kind == "formula":
                text, pace = formula_voice(s.spoken), FORMULA_PACE
            elif b.kind in ("text", "caption"):
                text, pace = phrase(s.spoken)
            else:
                text, pace = s.spoken, 1.0
            units.append(Unit(text, pause, b.kind, page, b.id, s.id, s.text, b.kind == "formula", pace))
    if tail is not None:
        units.append(Unit(tail["spoken"], "paragraph", "text", page, "", tail["id"], tail["text"]))
    return units
