"""
index.py - the per-page index behind seeking and highlighting.

A page is a list of Blocks in reading order (heading, text, caption, formula, table). Each block knows where it
is on the page and holds Sentences; a sentence keeps
  text    the original words as printed (the viewer finds and highlights them; "" for formulas and tables,
          which are highlighted as a whole box)
  spoken  what the voice says (citations dropped, symbols and numbers as words, formulas read out)
so a click at (page, x, y) can be turned into "start at this sentence" with a lookup, and the sentence being
spoken can be highlighted.

Coordinates are PDF points with the origin at the top-left of the page (x to the right, y down).
"""
import re
from dataclasses import asdict, dataclass, field

from pdf_reader.script import UNREADABLE, script_items
from pdf_reader.text_norm import normalize, normalize_numbers, prose, symbols

ABBREV = re.compile(r"(?:\b(?:e\.g|i\.e|et al|cf|vs|resp|w\.r\.t|i\.i\.d|r\.v|a\.s|etc|approx|Fig|Figs|Eq|Eqs|Sec|"
                    r"Secs|No|Thm|Prop|Def|Rem|Lem|Ref|Refs|Ch|Vol|pp|resp)|\b[A-Z])\.$")
BOUNDARY = re.compile(r"[.!?]['\")\]]*\s+(?=[A-Z0-9(\[\"'])")
SENTENCE_END = re.compile(r"[.!?:]['\")\]]*$")


@dataclass
class Sentence:
    id: str
    text: str
    spoken: str


@dataclass
class Block:
    id: str
    page: int
    kind: str
    bbox: tuple  # (left, top, right, bottom) in PDF points, top-left origin: the first of `boxes`
    sentences: list = field(default_factory=list)
    continues_from: str = ""  # id of the block on the previous page whose last sentence was joined in here
    # every box of the paragraph on this page, in reading order: a paragraph that goes on in the next column has
    # two (DoRA p2: the bottom of column 1 and the top of column 2); its text is searched and clicked in all
    boxes: list = field(default_factory=list)


def split_sentences(text):
    """Split printed text into sentences, not at "e.g.", "et al.", "Fig.", initials like "J."."""
    out, start = [], 0
    for m in BOUNDARY.finditer(text):
        head = text[start:m.start() + 1]
        if ABBREV.search(head.rstrip("'\")]")):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def _speak_text(sentence):
    return normalize(symbols(prose(sentence)))


def page_blocks(doc, pdf, speaker, state, stats, page_no):
    """Index one converted page. Uses the same rules as the whole-document script (script.script_items)."""
    height = doc.pages[page_no].size.height
    blocks = []
    for k, (kind, part, item) in enumerate(script_items(doc, pdf, speaker, state, stats)):
        boxes = []
        for prov in item.prov:
            if prov.page_no == page_no:
                bb = prov.bbox.to_top_left_origin(height)
                boxes.append((round(bb.l, 1), round(bb.t, 1), round(bb.r, 1), round(bb.b, 1)))
        bbox = boxes[0] if boxes else None
        bid = f"{page_no}.{k}"
        if kind in ("text", "caption"):
            pairs = [(s, _speak_text(s)) for s in split_sentences(" ".join(item.text.split()))]
        elif kind == "heading":
            pairs = [(" ".join(item.text.split()), normalize(part))]
        elif kind == "formula":
            pairs = [("", part if part == UNREADABLE else normalize_numbers(part))]
        else:  # table: read sentence by sentence, highlighted as one box
            pairs = [("", normalize(s)) for s in split_sentences(part)]
        sentences = [Sentence(f"{bid}.{i}", t, s) for i, (t, s) in enumerate([p for p in pairs if p[1].strip()])]
        if sentences:
            blocks.append(Block(bid, page_no, kind, bbox, sentences, boxes=boxes))
    return blocks


def take_tail(blocks):
    """The unfinished last sentence of a page (cut by the page break), removed from its block, with the block
    id and position (the block disappears if this was its only sentence, and the viewer still needs to know
    where those words are); None if the page ends with a finished sentence."""
    if not blocks or blocks[-1].kind != "text" or SENTENCE_END.search(blocks[-1].sentences[-1].text):
        return None
    block = blocks[-1]
    tail = block.sentences.pop()
    if not block.sentences:
        blocks.pop()
    return {"block": block.id, "bbox": block.bbox, "boxes": block_boxes(block), "sentence": asdict(tail)}


def join_tail(tail, blocks):
    """Join the previous page's unfinished sentence to the first body paragraph of this page (figure captions
    and tables at the top of a page are floats and are read after the finished sentence). True if joined."""
    lead = 0
    while lead < len(blocks) and blocks[lead].kind in ("caption", "table"):
        lead += 1
    if lead >= len(blocks) or blocks[lead].kind != "text":
        return False
    last, target = tail["sentence"]["text"], blocks[lead]
    first = target.sentences[0]
    if re.search(r"-\w+-$", last):  # "one-to-" + "many": the word's own hyphen
        joined = last + first.text
    elif re.search(r"\w-$", last):  # "meth-" + "ods": a line-break hyphen
        joined = last[:-1] + first.text
    else:
        joined = last + " " + first.text
    target.sentences[0] = Sentence(first.id, joined, _speak_text(joined))
    target.continues_from = tail["block"]
    if lead:
        blocks.insert(0, blocks.pop(lead))
    return True


def blocks_text(blocks):
    """The page as a plain script: one paragraph per block."""
    return "\n\n".join(" ".join(s.spoken for s in b.sentences) for b in blocks)


def to_json(blocks):
    return [asdict(b) for b in blocks]


def from_json(data):
    return [Block(**{**b, "bbox": tuple(b["bbox"]) if b["bbox"] else None,
                     "boxes": [tuple(x) for x in b.get("boxes") or []],
                     "sentences": [Sentence(**s) for s in b["sentences"]]}) for b in data]


def block_boxes(block):
    """All of a block's boxes (a block from an older cache has only its `bbox`)."""
    return list(block.boxes) or ([block.bbox] if block.bbox else [])


def in_box(box, x, y, margin=2):
    return box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin


def locate(blocks, x, y):
    """The block at (x, y) on its page (in any of its boxes), else the first block below that point in reading
    order: where reading should start after a click. Returns a Block or None."""
    for b in blocks:
        if any(in_box(box, x, y) for box in block_boxes(b)):
            return b
    below = [b for b in blocks if any(box[1] >= y for box in block_boxes(b))]
    same_column = [b for b in below if any(box[1] >= y and box[0] <= x <= box[2] for box in block_boxes(b))]
    return (same_column or below or [None])[0]
