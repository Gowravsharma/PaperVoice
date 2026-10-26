// phrasing.mjs - make a formula sound like one spoken phrase with breathing points, without changing its meaning.
//
// readable.mjs decides how notation is *named*; this module decides where a formula *breathes*. The speech
// engine's output has no punctuation, so a long formula reached the voice as one breathless 100-word sentence.
// speakPhrased(): a formula is read piece by piece at its top level - split at relations (=, :=, ≤, ∈, ↦),
// separating commas, and top-level + / − in long expressions, never inside brackets or |bars| - and the pieces
// are joined with pauses: a comma (the voice pauses ~0.16 s) before a relation or a long term, a semicolon
// (~0.30 s) at a separator. Each piece is a complete sub-expression, so the engine reads it exactly as before.
// Rejected on the way: pause tokens inserted into the MathML changed the engine's reading ("a plus b" became
// "positive b"); the engine's own SSML <break> marks have no notion of depth and fell inside small pieces
// ("X hat sub h, superscript t comma x"). The formula's final comma/full stop (sentence punctuation) is not read.
//
// Function arguments: "phi of open paren t comma x close paren" -> "phi of t comma x", but only when every
// argument is a plain symbol (letters, numbers, simple scripts, primes, the placeholder dot) - then the
// brackets carry no grouping information. (The engine's own Paren_Silent setting was rejected: it also drops
// absolute values, norms, inner products and interval ends, which changes the meaning.)
//
// finish(speech): pause tokens -> punctuation, and wording for symbols with a single meaning:
//   ":=" "colon equals" -> "is defined as", "∂" "partial differential" -> "partial", "‖x‖" "the metric of" ->
//   "the norm of", "↦" "long right arrow from bar" -> "maps to", a limit's "over h right arrow 0" -> "as h
//   approaches 0".

const REL = 3, BIN = 2, CLOSE = 5, PUNCT = 6;
const BIG = new Set(["∑", "∫", "∬", "∮", "∏", "⋃", "⋂", "lim", "sup", "inf", "max", "min", "lim sup", "lim inf",
  "limsup", "liminf", "ess sup", "ess inf"]);
const LIMITS = new Set(["lim", "lim sup", "lim inf", "limsup", "liminf"]);
const ARROWS = new Set(["→", "⟶", "↘", "↗", "↓", "↑", "⇀"]);

const word = (f, s) => {
  const t = f.create("mtext");
  t.appendChild(f.create("text").setText(s));
  return t;
};
const tok = (n) => (n && n.isToken ? n.getText() : "");
function leaves(n, out = []) {
  if (!n) return out;
  if (n.isToken) out.push(n);
  else for (const c of n.childNodes) leaves(c, out);
  return out;
}
const size = (n) => leaves(n).filter((l) => l.kind !== "mspace" && tok(l).trim()).length;
function core(n) {
  while (n && ["TeXAtom", "mrow", "inferredMrow"].includes(n.kind) && n.childNodes.length === 1) n = n.childNodes[0];
  return n;
}
// ------------------------------------------------------------------ function arguments
const ATOM_MO = new Set([",", "′", "″", "'", "⋅", "·"]);
function isAtom(n) {
  // a plain argument: identifiers, numbers, simple scripts, primes, the placeholder dot (mtext "dot")
  return leaves(n).every((l) => ["mi", "mn"].includes(l.kind) || (l.kind === "mo" && ATOM_MO.has(tok(l)))
    || (l.kind === "mtext" && tok(l) === "dot"));
}
function isFunctionName(n) {
  const c = core(n);
  if (!c) return false;
  if (c.kind === "mi") return /^\p{L}$/u.test(tok(c)) || tok(c).length > 1;
  return ["msub", "msup", "msubsup", "mover", "munder", "mrow"].includes(c.kind) && leaves(c)[0]?.kind === "mi";
}
function endsCleanly(kids, j) {
  // what follows the closing bracket must not be a factor: "mu of t comma x h" would hide that h multiplies
  const next = kids[j + 1];
  if (!next) return true;
  const c = core(next);
  return c && c.kind === "mo" && [REL, BIN, PUNCT, CLOSE].includes(c.texClass);
}
function applications(node, out = []) {
  // every "name(...)" in the formula: {row, i (index of the bracket), flat, end, name, ok (brackets can go)}
  if (node.isToken) return out;
  for (const c of node.childNodes) applications(c, out);
  const kids = node.childNodes;
  for (let i = 1; i < kids.length; i++) {
    const fn = kids[i - 1];
    if (!isFunctionName(fn)) continue;
    const name = leaves(fn).map(tok).join("");
    if (kids[i].kind === "mo" && tok(kids[i]) === "(") { // \varphi(t,x): "(" ... ")" as siblings
      let j = i + 1;
      while (j < kids.length && !(kids[j].kind === "mo" && tok(kids[j]) === ")")) j++;
      const args = kids.slice(i + 1, j);
      const ok = j < kids.length && args.length > 0 && args.every(isAtom) && args.some((a) => tok(a) === ",")
        && endsCleanly(kids, j);
      out.push({ row: node, i, flat: true, end: j, name, ok });
    } else if (["mrow", "TeXAtom"].includes(kids[i].kind)) { // F\left(\cdot, v\right): one bracketed group
      const g = core(kids[i]).kind === "mrow" ? core(kids[i]) : kids[i];
      const inner = g.childNodes;
      if (inner.length >= 3 && tok(inner[0]) === "(" && tok(inner[inner.length - 1]) === ")") {
        const args = inner.slice(1, -1);
        const ok = args.every(isAtom) && args.some((a) => tok(a) === ",") && endsCleanly(kids, i);
        out.push({ row: node, i, flat: false, group: g, name, ok });
      }
    }
  }
  return out;
}

function silenceArguments(math, f) {
  // A function's brackets go only if ALL its uses in the formula can lose them: the engine decides "sigma of
  // (t, x)" vs "sigma times (t, x)" from how sigma is used elsewhere, so rewriting some uses changed others.
  const apps = applications(math);
  const blocked = new Set(apps.filter((a) => !a.ok).map((a) => a.name));
  const todo = apps.filter((a) => a.ok && !blocked.has(a.name)).sort((x, y) => y.i - x.i); // right to left
  for (const a of todo) {
    const kids = a.row.childNodes, fn = kids[a.i - 1];
    // one unit (function, "of", arguments): its commas must not look like top-level separators. The "of" is
    // explicit: relying on the engine to infer the function application lost it ("F tilde dot comma u")
    if (a.flat) {
      const group = f.create("mrow", {}, [fn, word(f, "of"), ...kids.slice(a.i + 1, a.end)]);
      kids.splice(a.i - 1, a.end - a.i + 2, group);
      group.parent = a.row;
    } else {
      a.group.childNodes.pop();
      a.group.childNodes.shift();
      const group = f.create("mrow", {}, [fn, word(f, "of"), kids[a.i]]);
      kids.splice(a.i - 1, 2, group);
      group.parent = a.row;
    }
  }
}
// ------------------------------------------------------------------ limits
function bigOperator(k) {
  // \sum_{..}^{..}, \sup_{..}, \lim_{..}: an operator with limits; returns {script} or null
  if (!["munderover", "munder", "msubsup", "msub", "msup", "mover"].includes(k.kind)) return null;
  const base = core(k.childNodes[0]);
  const name = leaves(base).map(tok).join("").replace(/\s+/g, " ").trim();
  if (!BIG.has(name)) return null;
  return { name, scripts: k.childNodes.slice(1) };
}

function limitArrows(node, f) {
  // the arrow under a limit means "approaches": lim_{h -> 0}
  if (node.isToken) return;
  for (const c of node.childNodes) limitArrows(c, f);
  const big = bigOperator(node);
  if (!big || !LIMITS.has(big.name)) return;
  for (const s of big.scripts) {
    for (const l of leaves(s)) {
      if (l.kind === "mo" && ARROWS.has(tok(l)) && l.parent) {
        const kids = l.parent.childNodes;
        const w = word(f, "approaches");
        kids[kids.indexOf(l)] = w;
        w.parent = l.parent;
      }
    }
  }
}

export function phrase(math, f) {
  // only rewrites that keep the engine's reading of the structure intact; breathing points come from the
  // engine's own structure marks (see breathe()), because extra tokens in the tree changed its reading
  // ("a plus b" became "positive b")
  silenceArguments(math, f);
  limitArrows(math, f);
  return math;
}

// ------------------------------------------------------------------ breathing points: split at the top level
const OPEN = 4;
const BARS = new Set(["|", "‖", "∣", "∥"]);
const TERM_OPS = { "+": "plus", "−": "minus", "-": "minus", "±": "plus or minus", "∓": "minus or plus" };

function topRow(math) {
  let n = math;
  while (n && !n.isToken && n.childNodes.length === 1 && ["math", "inferredMrow", "mrow", "TeXAtom"].includes(n.kind)) {
    n = n.childNodes[0];
  }
  return n && !n.isToken && n.childNodes.length > 1 ? n : null;
}

function barsIn(k) {
  // absolute-value / norm bars sitting directly in the row (|x|, or the closing bar of |x|_\infty)
  if (k.kind === "mo") return BARS.has(tok(k)) ? 1 : 0;
  if (["msub", "msup", "msubsup"].includes(k.kind)) return BARS.has(tok(core(k.childNodes[0]))) ? 1 : 0;
  return 0;
}

function pieces(row) {
  // [{nodes}, {op, kind: "rel" | "term" | "sep"}, {nodes}, ...] at depth 0 only (not inside brackets or bars)
  const kids = row.childNodes, out = [];
  let cur = [], depth = 0, bars = 0;
  const total = kids.reduce((s, k) => s + Math.max(1, size(k)), 0);
  let seen = 0;
  for (let i = 0; i < kids.length; i++) {
    const k = kids[i];
    const top = depth === 0 && bars % 2 === 0;
    const curSize = cur.reduce((s, n) => s + Math.max(1, size(n)), 0);
    if (top && k.kind === "mo" && k.texClass === REL && cur.length) {
      out.push({ nodes: cur }, { op: k, kind: "rel" }); cur = [];
    } else if (top && k.kind === "mo" && TERM_OPS[tok(k)] && cur.length && curSize >= 6 && total - seen >= 5) {
      out.push({ nodes: cur }, { op: k, kind: "term" }); cur = [];
    } else if (top && k.kind === "mo" && k.texClass === PUNCT && [",", ".", ";"].includes(tok(k))) {
      const next = kids[i + 1];
      if (!next) { seen += 1; continue; } // the sentence's own punctuation at the end: not read out
      const spaceOrText = leaves(next).every((l) => l.kind === "mspace" || l.kind === "mtext"); // \quad, \text{on}
      if (cur.length && (spaceOrText || curSize >= 8)) {
        out.push({ nodes: cur }, { op: k, kind: "sep" }); cur = [];
      } else cur.push(k);
    } else {
      cur.push(k);
    }
    if (k.kind === "mo" && k.texClass === OPEN) depth++;
    if (k.kind === "mo" && k.texClass === CLOSE) depth = Math.max(0, depth - 1);
    bars += barsIn(k);
    seen += Math.max(1, size(k));
  }
  if (cur.length) out.push({ nodes: cur });
  return out;
}

function lcsPairs(a, b) {
  // matched index pairs of the longest common subsequence of two word lists
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  }
  const pairs = [];
  for (let i = 0, j = 0; i < n && j < m;) {
    if (a[i] === b[j]) pairs.push([i++, j++]);
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  return pairs;
}

export function speakPhrased(math, speakNodes, speakWhole) {
  // The words come from reading the WHOLE formula once (the engine decides "delta of t" vs "delta times t"
  // from the whole formula, so reading pieces alone changed words in 4 of 199 formulas). The pieces are read
  // only to find where the top-level boundaries fall; those positions are matched onto the whole reading and
  // the pauses inserted there: a comma before a relation (left side of 4+ words) or a long term, a semicolon
  // at a separating comma. Formulas without such a top level (tables, aligned rows) get no extra pauses here.
  const whole = finish(speakWhole(math)).replace(/(?:,? (?:comma|period))+$/, ""); // final sentence punctuation
  const row = topRow(math);
  const tabular = (n) => leaves(n).length && !n.isToken && (n.kind === "mtable" || n.childNodes.some(tabular));
  const parts = row && !row.childNodes.some(tabular) ? pieces(row) : null;
  if (!parts || parts.length < 3) return whole;

  const seq = [], marks = new Map(); // piece words in order; marks: piece-word index after which a pause goes
  let lastLen = 0;
  for (const p of parts) {
    if (!p.nodes && seq.length) {
      if (p.kind === "rel" && lastLen >= 4) marks.set(seq.length - 1, ","); // "f of x equals" needs no pause
      if (p.kind === "term") marks.set(seq.length - 1, ",");
      if (p.kind === "sep") { marks.set(seq.length - 1, ";"); continue; } // the separator comma is not read
    }
    const words = finish(speakNodes(p.nodes || [p.op])).split(/\s+/).filter(Boolean);
    seq.push(...words);
    if (p.nodes) lastLen = words.length;
  }
  const norm = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}-]/gu, "");
  const out = whole.split(/\s+/);
  const at = new Map(lcsPairs(seq.map(norm), out.map(norm))); // piece-word index -> whole-word index
  for (const [k, punct] of marks) {
    let i = k; // the last matched word at or before the boundary; no pause if the match is not close
    while (i >= 0 && !at.has(i)) i--;
    if (i < 0 || k - i > 2) continue;
    const j = at.get(i);
    if (j < out.length - 1 && !/[,;]$/.test(out[j])) out[j] += punct;
  }
  return out.join(" ");
}

export function finish(speech) {
  let s = ` ${speech} `;
  s = s.replace(/\bcolon equals\b/g, "is defined as")
    .replace(/\bpartial differential\b/g, "partial")
    .replace(/\bthe metric of\b/g, "the norm of").replace(/\bend metric\b/g, "end norm")
    .replace(/\b(?:long )?right arrow from bar\b/g, "maps to")
    .replace(/\b(lim(?: sup| inf)?|limit(?: superior| inferior)?) over (.{1,40}?) approaches\b/g, "$1 as $2 approaches")
    // rows of multi-line layouts: a clause pause before each "Line 2:" / "Case 2:" (no words change)
    .replace(/\s+((?:Line|Case|Row) \d+:)/g, "; $1").replace(/\b(\d+ (?:lines|cases|rows))\s*;/g, "$1,");
  s = s.replace(/\s*([,;])\s*(?:[,;]\s*)+/g, "$1 ").replace(/^[\s,;]+|[\s,;]+$/g, "").replace(/\s+/g, " ");
  return s;
}
