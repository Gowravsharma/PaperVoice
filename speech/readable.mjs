// readable.mjs - rewrite a MathJax MathML tree so ClearSpeak reads research-paper notation naturally.
//
// ClearSpeak treats every superscript as a power ("the h-th power of v"). In papers most letter
// superscripts are indices. Rules (applied before speech):
//   index        v^h, X_h^{t,x}, X^{(2)}, w^\varepsilon  -> "v superscript h"
//                (superscript made of letters/commas, or wrapped in parentheses, on a variable)
//   transpose    A^T, \sigma^{\mathrm T}                -> "A transpose"
//   fraction     h^{-1/2}                               -> "h to the power negative 1 over 2"
//   blackboard   \mathbb{R}^d, \mathbb{E}[...]          -> "R superscript d", "E of ..."
//                (plain <mi> letter; an <mtext> letter would be read "cap R")
//   placeholder  F(\cdot, v), v(T,\cdot)                -> "dot"
// Numeric powers (\sigma^2), e^{...}, primes (x') and fenced bases (|x|^q) keep ClearSpeak's reading.
// A rewritten Latin-letter script followed by "(" gets an explicit function application (U+2061),
// because the new <mrow> hides the variable from ClearSpeak's "v of (t, x)" heuristic.

const BLACKBOARD = { "ℝ": "R", "ℕ": "N", "ℤ": "Z", "ℚ": "Q", "ℂ": "C", "ℙ": "P" };
const DOUBLE_STRUCK_BASE = 0x1d538; // MATHEMATICAL DOUBLE-STRUCK CAPITAL A

function plainLetter(ch) {
  if (BLACKBOARD[ch]) return BLACKBOARD[ch];
  const cp = ch.codePointAt(0);
  if (cp >= DOUBLE_STRUCK_BASE && cp < DOUBLE_STRUCK_BASE + 26) return String.fromCharCode(65 + cp - DOUBLE_STRUCK_BASE);
  return null;
}

const text = (f, s) => {
  const t = f.create("mtext");
  t.appendChild(f.create("text").setText(s));
  return t;
};
const row = (f, kids) => f.create("mrow", {}, kids);

// unwrap TeXAtom / inferred mrow / single-child mrow
function core(n) {
  while (n && (n.kind === "TeXAtom" || n.kind === "mrow" || n.kind === "inferredMrow") && n.childNodes.length === 1) {
    n = n.childNodes[0];
  }
  return n;
}

function leaves(n, out = []) {
  if (!n) return out;
  if (n.isToken) out.push(n);
  else for (const c of n.childNodes) leaves(c, out);
  return out;
}

const tokText = (n) => (n && n.isToken ? n.getText() : "");
const isBlackboard = (n) => n && n.kind === "mi" && (plainLetter(n.getText()) !== null ||
  (n.attributes.get("mathvariant") === "double-struck" && /^[A-Z]$/.test(n.getText())));
const blackboardLetter = (n) => plainLetter(n.getText()) || n.getText();

// the variable a script hangs on: X in \hat X, X in X_h
function baseVariable(n) {
  n = core(n);
  while (n && ["mover", "munder", "msub", "TeXAtom", "mrow", "inferredMrow"].includes(n.kind) && n.childNodes.length) {
    n = core(n.childNodes[0]);
  }
  return n;
}

function supKind(base, sup) {
  const b = baseVariable(base);
  const ls = leaves(sup);
  const s = ls.map(tokText).join("");
  if (s === "T" || s === "⊤") return "transpose"; // also on a closing bracket: \sigma(t,x)^T
  if (/^[′″‴'*∗]+$/.test(s)) return "power"; // x', v^*: ClearSpeak already says "prime", "star"
  if (/^[+\-−]$/.test(s)) return "index"; // F_\gamma^-, m_F^-: positive / negative part, not a power
  // fractional exponents on any base, (i+2.5)^{-5/2} too ("raised to the negative 5 divided by 2 power")
  if (ls.some((l) => l.kind === "mo" && l.getText() === "/") || core(sup)?.kind === "mfrac") return "fraction";
  if (!b || b.kind !== "mi") return "power"; // fences, numbers, closing brackets: |x|^q, (a+b)^2
  if (b.getText() === "e") return "power";
  if (ls.length && ls.every((l) => l.kind === "mi" || (l.kind === "mo" && l.getText() === ","))) return "index";
  if (ls.length > 2 && tokText(ls[0]) === "(" && tokText(ls[ls.length - 1]) === ")") return "index";
  return "power";
}

function rewrite(node, f) {
  for (const c of [...node.childNodes]) rewrite(c, f);

  if (node.kind === "msup" || node.kind === "msubsup") {
    const base = node.childNodes[node.base];
    const sup = node.childNodes[node.sup];
    const sub = node.kind === "msubsup" ? node.childNodes[node.sub] : null;
    const kind = supKind(base, sup);
    if (kind === "power") return;
    const head = sub ? f.create("msub", {}, [base, sub]) : base;
    const tail = {
      index: [text(f, "superscript"), sup],
      transpose: [text(f, "transpose")],
      fraction: [text(f, "to the power"), sup],
    }[kind];
    const neu = row(f, [head, ...tail]);
    replace(node, neu);
    const b = baseVariable(base);
    if (kind === "index" && b && /^[A-Za-z]$/.test(b.getText()) && startsWithParen(nextSibling(neu))) {
      const apply = f.create("mo");
      apply.appendChild(f.create("text").setText("⁡"));
      neu.parent.childNodes.splice(neu.parent.childNodes.indexOf(neu) + 1, 0, apply);
      apply.parent = neu.parent;
    }
    return;
  }

  // MathJax marks symbols such as ∞ as upright (mathvariant=normal) and ClearSpeak then says "normal infinity";
  // "normal" is only meaningful for letters (\mathrm{T})
  if (node.kind === "mi" && node.attributes.get("mathvariant") === "normal" && !/^[A-Za-z]+$/.test(node.getText())) {
    delete node.attributes.getAllAttributes().mathvariant; // Attributes has no unset()
  }

  if (isBlackboard(node)) {
    const mi = f.create("mi");
    mi.appendChild(f.create("text").setText(blackboardLetter(node)));
    return replace(node, mi);
  }

  if (node.kind === "mo" && (node.getText() === "⋅" || node.getText() === "·")) {
    const sibs = node.parent ? node.parent.childNodes : [];
    const i = sibs.indexOf(node);
    const prev = tokText(core(sibs[i - 1])), next = tokText(core(sibs[i + 1]));
    if (["(", ",", "["].includes(prev) || i === 0) {
      if ([",", ")", "]"].includes(next) || i === sibs.length - 1) return replace(node, text(f, "dot"));
    }
  }
}

function nextSibling(n) {
  const sibs = n.parent ? n.parent.childNodes : [];
  return sibs[sibs.indexOf(n) + 1];
}

function startsWithParen(n) {
  const first = leaves(n)[0];
  return tokText(first) === "(";
}

function replace(old, neu) {
  const p = old.parent;
  if (p) p.replaceChild(neu, old);
}

export function makeReadable(math, factory) {
  rewrite(math, factory);
  return math;
}
