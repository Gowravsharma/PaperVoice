r"""
table_speech.py - read a Docling table aloud (v1: literal, rule-based, no models).

speak_table(table, doc) -> SpokenTable(text, consumed)

Reading order for a listener who cannot look back:
  "Table 3. <first sentence of caption>. It has 7 rows and 4 columns: sigma, gamma, K, DTW.
   GT: sigma 54.4, gamma 0.836, K 0.977, DTW no value. ... End of table 3."
  - every value is read with its column name; each row starts with its first cell (the row label)
  - long tables (> max_rows) read the first rows and say how many follow, unless verbose=True

Refinements of Docling's grid:
  - captions: Docling often leaves "(a) ...", "(b) ..." and "Table N: ..." as plain text right after the
    table. They are picked up here and returned in `consumed` so the body reader can skip them.
  - side-by-side tables: k sub-captions and a column count divisible by k -> split into k parts.
  - misaligned rows: TableFormer sometimes stretches one cell over two text lines and pushes the rest of
    its text into the next row (FastSpeech 2, Table 1). If a cell is much taller than its row and overlaps
    the next row, the cell texts of those rows are re-read from the PDF text layer inside
    (column x-span) x (row y-band). Needs `pdf` (a pypdfium2 PdfDocument); without it such rows are skipped.
  - a part whose body repeats one value in a whole column (e.g. "Method" in every row) or has no column
    names is reported as unreliable instead of being read out with wrong numbers.

Cell text is normalised literally (v1: symbols are read as written, no meaning from context):
  ±, ×, %, leading signs, Greek letters, units in column names ("(ms)"), "× 10-1" flattened exponents,
  joined words ("fromMFA"), citations dropped, empty markers ("/", "-") read as "no value".
"""
import re
from dataclasses import dataclass, field

from docling_core.types.doc import DoclingDocument, NodeItem, TableItem

from pdf_reader.pdfium_safe import text_page
from pdf_reader.text_norm import RELATIONS as TEXT_RELATIONS
from pdf_reader.text_norm import symbols

# exact-case, unambiguous units only; applied to column names, never to captions ("kurtosis (K)")
UNITS = {"ms": "milliseconds", "s": "seconds", "sec": "seconds", "min": "minutes", "h": "hours",
         "Hz": "hertz", "kHz": "kilohertz", "dB": "decibels", "%": "percent", "GB": "gigabytes",
         "MB": "megabytes"}
EMPTY = {"/", "-", "–", "—", "n/a", "N/A", "×"}
RELATIONS = {**TEXT_RELATIONS, "<": "less than", ">": "greater than"}  # bare < > only inside table cells
ABBREV = re.compile(r"(?:\bvs|\be\.g|\bi\.e|\bet al|\bFig|\bEq|\bNo|\bcf|\betc)\.$")
SUBCAPTION = re.compile(r"^\(([a-z])\)\s*(.+)")
CAPTION = re.compile(r"^Table\s+(\d+)\s*[:.]\s*(.*)", re.S)
CITATION = re.compile(r"\s*\((?:[A-Z][^()]*?et al\.?|[A-Z][a-z]+(?: and [A-Z][a-z]+)?),?\s*\d{4}[a-z]?\)")


@dataclass
class SpokenTable:
    text: str
    consumed: list[NodeItem] = field(default_factory=list)  # caption items read here; skip them in the body


def say(text, units=False, cell=False):
    """Literal, speakable form of one cell or caption. units=True also expands "(ms)" etc. (column names).
    cell=True: a slash between numbers is read "slash" - in a table "2/3" is usually two values (a ratio, a
    score pair, "2/2s" versions), so it is not turned into a fraction ("two thirds") or a date."""
    t = " ".join(text.split())
    if t in EMPTY:
        return "no value"
    t = CITATION.sub("", t)
    if cell:
        t = re.sub(r"(?<=\d)\s*/\s*(?=\d)", " slash ", t)
    if units:
        t = re.sub(r"\((%|[A-Za-z]{1,3})\)", lambda m: "in " + UNITS[m.group(1)] if m.group(1) in UNITS
                   else m.group(0), t)
    t = re.sub(r"([a-z])([A-Z]{2,})", r"\1 \2", t)  # fromMFA -> from MFA
    # the text layer flattens 10^{-1} to "10-1" / "10 − 1"; only after × or a relation, so ranges like
    # "10-12" are left alone
    t = re.sub(r"([×x≪≫≤≥<>≈=]\s*)10\s*[-−]\s*(\d+)\b", r"\g<1>10 to the power minus \2", t)
    t = re.sub(r"[×x]\s*(10 to the power)", r"times \1", t)
    t = re.sub(r"(\d)\s*[×x](?![\w])", r"\1 times", t)  # 38.3x speed-up
    t = re.sub(r"(?<![\w)])[-−–](?=\d)", "minus ", t)
    t = re.sub(r"(?<![\w)])\+(?=\d)", "plus ", t)
    t = re.sub(r"\s[-−–]\s", " minus ", t)
    t = re.sub(r"\s?\+\s?(?=[A-Za-z(])|\s\+\s", " plus ", t)  # "Linear +WaveNet", "Mel + PWG"
    t = symbols(t, RELATIONS)
    t = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", t))
    return " ".join(t.split())


def find_captions(table: TableItem, doc: DoclingDocument):
    """(number, caption, [subcaptions], consumed items) for a table."""
    caption = table.caption_text(doc).strip()
    items = [it for it, _ in doc.iterate_items()]
    i = items.index(table)
    page = table.prov[0].page_no if table.prov else None
    subs, consumed = [], []
    for it in items[i + 1:i + 8]:
        txt = (getattr(it, "text", "") or "").strip()
        if not it.prov or it.prov[0].page_no != page:
            break
        if m := SUBCAPTION.match(txt):
            subs.append(m.group(2))
            consumed.append(it)
        elif CAPTION.match(txt) and not CAPTION.match(caption):
            caption = txt
            consumed.append(it)
            break
        else:
            break
    m = CAPTION.match(caption)
    number = m.group(1) if m else None
    text = m.group(2) if m else caption
    return number, text, subs, consumed


def _repair_rows(table: TableItem, doc: DoclingDocument, grid, pdf):
    """Re-read misaligned rows from the PDF text layer. Returns (grid, rows that stay unreadable)."""
    cells = table.data.grid
    tall = {}  # row -> column of the stretched cell
    for r, row in enumerate(cells[:-1]):
        hs = [c.bbox.b - c.bbox.t for c in row if c.bbox and c.text.strip()]
        if not hs:
            continue
        line = min(hs)
        nxt = [c.bbox.t for c in cells[r + 1] if c.bbox and c.text.strip()]
        for col, c in enumerate(row):
            if c.bbox and c.text.strip() and c.bbox.b - c.bbox.t > 1.6 * line and nxt and c.bbox.b > min(nxt):
                tall.setdefault(r, col)
                tall.setdefault(r + 1, col)
    if not tall:
        return grid, set()
    if pdf is None:
        return grid, tall
    chars = []  # (x centre, y centre from top, char) - a char belongs to a cell if its centre is inside
    with text_page(pdf, table.prov[0].page_no) as (page, tp):
        height = page.get_height()
        for i in range(tp.count_chars()):
            left, bot, right, top_ = tp.get_charbox(i)
            chars.append(((left + right) / 2, height - (bot + top_) / 2, tp.get_text_range(i, 1)))
    ncols = len(grid[0])
    spans = []
    for c in range(ncols):
        boxes = [row[c].bbox for row in cells if row[c].bbox and row[c].text.strip()]
        spans.append((min(b.l for b in boxes), max(b.r for b in boxes)) if boxes else None)
    grid = [list(r) for r in grid]
    for r, col0 in sorted(tall.items()):
        cand = [(col, c.bbox) for col, c in enumerate(cells[r]) if c.bbox and c.text.strip()]
        line = min(b.b - b.t for _, b in cand)
        # the row band comes from the single-line cell nearest the stretched column: cells from other
        # (possibly broken) parts of a side-by-side table can sit at a different height
        _, ref = min(((abs(col - col0), b) for col, b in cand if b.b - b.t <= 1.6 * line), key=lambda x: x[0])
        top, bottom = ref.t, ref.b
        for c, span in enumerate(spans):
            if span:
                txt = "".join(ch for x, y, ch in chars
                              if span[0] - 1 <= x <= span[1] + 1 and top - 1 <= y <= bottom + 1)
                grid[r][c] = " ".join(txt.split())
    return grid, set()


def _first_sentence(text):
    """Up to the first full stop that ends a sentence (not "vs.", "e.g.", "et al.")."""
    for m in re.finditer(r"[.!?](?=\s+[A-Z(]|\s*$)", text):
        if not ABBREV.search(text[:m.end()]):
            return text[:m.end()].strip()
    return text.strip()


def _split_shared(labels):
    """Spanning header over two or more value columns -> (shared, [labels with it removed]).
    Columns outside the span (e.g. a trailing "p value") keep their own label."""
    parts = [[p.strip() for p in l.split(",")] for l in labels]
    firsts = [p[0] for p in parts if len(p) >= 2]
    if not firsts:
        return "", labels
    shared = max(set(firsts), key=firsts.count)
    if firsts.count(shared) < 2:
        return "", labels
    return shared, [", ".join(p[1:]) if len(p) >= 2 and p[0] == shared else ", ".join(p) for p in parts]


def _row_groups(body):
    """Detect a first column of stacked words ("North" / "American" / "English" / "" / ...) that really is one
    row-spanning group label. Returns [(group label or None, rows)]."""
    labels = [r[0] for r in body]
    empty = sum(1 for l in labels if not l)
    words = [l for l in labels if l]
    if not (empty >= len(labels) / 3 and words and all(re.fullmatch(r"[A-Za-z]+", w) for w in words)):
        return [(None, body)]
    groups, name, rows, prev_empty = [], [], [], False
    for r in body:
        if r[0] and (prev_empty or not rows):  # a label after an empty run starts a new group
            if rows:
                groups.append((" ".join(name), rows))
            name, rows = [r[0]], []
        elif r[0]:
            name.append(r[0])
        rows.append([""] + r[1:])  # the label words belong to the group, not to single rows
        prev_empty = not r[0]
    groups.append((" ".join(name), rows))
    return groups


def _labels(header_rows, cols):
    out = []
    for c in cols:
        parts = []
        for row in header_rows:
            if row[c] and row[c] not in parts:
                parts.append(row[c])
        out.append(", ".join(parts))
    return out


def _unreliable(labels, body):
    if not any(labels[1:]):
        return True
    for c in range(len(labels)):
        col = [r[c] for r in body if r[c]]
        if len(col) >= 3 and len(set(col)) == 1 and not re.search(r"\d", col[0]):
            return True  # a header word repeated down a whole column: structure is broken
    return False


def _read_part(labels, body, max_rows, verbose):
    shared, value_labels = _split_shared(labels[1:])
    spoken = [say(l, units=True) for l in value_labels]
    row_head = say(labels[0], units=True)
    out = [f"It has {len(body)} row{'s' * (len(body) != 1)} and {len(value_labels)} "
           f"column{'s' * (len(value_labels) != 1)}"
           + (f" under {say(shared, units=True)}" if shared else "")
           + f": {', '.join(l for l in spoken if l)}."]
    budget = len(body) if verbose or len(body) <= max_rows else 3
    for group, rows in _row_groups(body):
        if budget <= 0:
            break
        if group:
            out.append(f"{say(group)}.")
        for row in rows[:budget]:
            vals = [f"{spoken[c - 1]} {say(v, cell=True)}".strip() for c, v in enumerate(row) if c and v]
            label = say(row[0], cell=True)
            if label and re.fullmatch(r"[\d.,\s]+", row[0]) and row_head:
                label = f"{row_head} {label}"  # "30" alone means nothing: "Total layers 30"
            out.append((f"{label}: " if label else "") + ", ".join(vals) + ".")
        budget -= len(rows)
    if budget < 0 or (not verbose and len(body) > max_rows):
        out.append(f"And {len(body) - 3} more rows.")
    return out


def speak_table(table: TableItem, doc: DoclingDocument, pdf=None, max_rows: int = 12,
                verbose: bool = False) -> SpokenTable:
    number, caption, subs, consumed = find_captions(table, doc)
    grid = [[" ".join(c.text.split()) for c in row] for row in table.data.grid]
    if not grid:
        return SpokenTable("", consumed)
    grid, unreadable = _repair_rows(table, doc, grid, pdf)
    n_head = 0
    while n_head < len(grid) - 1 and any(c.column_header for c in table.data.grid[n_head]):
        n_head += 1
    n_head = max(n_head, 1)
    head = grid[:n_head]
    body = [r for i, r in enumerate(grid) if i >= n_head and i not in unreadable and any(r)]
    ncols = len(grid[0])
    cells = [v for r in body for v in r[1:] if v]
    if not caption and not subs and sum(bool(re.search(r"\d", v)) for v in cells) < len(cells) / 5:
        # no caption and (almost) no numbers: a layout false positive such as an author block
        return SpokenTable("", consumed)

    name = f"Table {number}" if number else "Table"
    out = [f"{name}." + (f" {say(_first_sentence(caption))}" if caption else "")]
    k = len(subs)
    if k > 1 and ncols % k == 0 and ncols // k >= 2:
        w = ncols // k
        out.append(f"It has {k} parts.")
        for p in range(k):
            cols = list(range(p * w, (p + 1) * w))
            labels = _labels(head, cols)
            part = [[r[c] for c in cols] for r in body]
            part = [r for r in part if any(r)]
            out.append(f"Part {chr(97 + p)}, {say(_first_sentence(subs[p]))}")
            if _unreliable(labels, part):
                out.append("This part could not be read reliably.")
            else:
                out.extend(_read_part(labels, part, max_rows, verbose))
    else:
        for s in subs:
            out.append(say(_first_sentence(s)))
        labels = _labels(head, range(ncols))
        if _unreliable(labels, body):
            out.append("This table could not be read reliably.")
        else:
            out.extend(_read_part(labels, body, max_rows, verbose))
    out.append(f"End of {name.lower() if not number else name}.")
    return SpokenTable(" ".join(out), consumed)
