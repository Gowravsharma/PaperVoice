// tex2speech.mjs - turn lines containing $...$ LaTeX into plain spoken text for TTS.
// Pipeline per formula: LaTeX --(MathJax)--> MathML --(Speech Rule Engine)--> words.
// Usage: node tex2speech.mjs [clearspeak|mathspeak] [--literal] [--no-phrasing] [--serve] < lines.txt
//   --serve keeps running and answers each input line as it arrives (used for page-by-page streaming).
//   The MathML is first rewritten by readable.mjs (indices, transpose, blackboard letters, placeholder dots),
//   then phrasing.mjs adds breathing points and fixes wording (--no-phrasing skips that);
//   --literal skips both and gives the engine's own reading.
import { createRequire } from "module";
import { makeReadable } from "./readable.mjs";
import { phrase, speakPhrased } from "./phrasing.mjs";
const require = createRequire(import.meta.url);

const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const { SerializedMmlVisitor } = require("mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js");
const sre = require("speech-rule-engine");
const { STATE } = require("mathjax-full/js/core/MathItem.js");

const args = process.argv.slice(2);
const literal = args.includes("--literal");
const phrasing = !literal && !args.includes("--no-phrasing"); // breathing points + wording (phrasing.mjs)
const domain = args.find((a) => !a.startsWith("--")) || "clearspeak";
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: AllPackages.filter((p) => p !== "bussproofs") });
const doc = mathjax.document("", { InputJax: tex });
const visitor = new SerializedMmlVisitor();

// Recognizer output is sometimes malformed TeX (unmatched \left, stray &, unbalanced braces). MathJax then
// returns an <merror> and the speech engine says nothing. These repairs only remove or balance markup -
// they never add content - and are tried cumulatively until the formula parses.
function balanceBraces(s) {
  let depth = 0, out = "";
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === "\\" && i + 1 < s.length) { out += c + s[++i]; continue; }
    if (c === "{") depth++;
    if (c === "}") { if (depth === 0) continue; depth--; }
    out += c;
  }
  return out + "}".repeat(depth);
}
const REPAIRS = [
  (s) => s.replace(/\\(left|right)\s*(?=[a-z]{2,}\b)/g, "\\$1\\").replace(/\\(left|right)\b\s*([.|()[\]]|\\[{}|]|\\[a-zA-Z]+)?/g, (_, lr, d) => d && d !== "." ? d : ""),
  (s) => s.replace(/\\begin\s*\{?[a-z]*\*?\}?(\{[^{}]*\})?|\\end\s*\{?[a-z]*\*?\}?/g, " ").replace(/&|\\\\/g, " "),
  balanceBraces,
  (s) => s.replace(/([_^])\s*(?=[}_^]|$)/g, "").replace(/\^(\{[^{}]*\}|\w)\s*\^/g, "^$1{}^"),
];
const hasError = (node) => { let e = false; node.walkTree((n) => { if (n.kind === "merror") e = true; }); return e; };

function tex2tree(latex) {
  let node = doc.convert(latex, { display: false, end: STATE.CONVERT });
  for (let i = 0; hasError(node) && i < REPAIRS.length; i++) {
    latex = REPAIRS[i](latex);
    node = doc.convert(latex, { display: false, end: STATE.CONVERT });
  }
  if (hasError(node)) throw new Error("unparsable TeX");
  if (!literal) makeReadable(node, doc.mmlFactory);
  if (phrasing) phrase(node, doc.mmlFactory);
  return node;
}

await sre.setupEngine({ locale: "en", domain, modality: "speech", style: "default" });
await sre.engineReady();

function speakFormula(latex) {
  const tree = tex2tree(latex);
  if (!phrasing) return sre.toSpeech(visitor.visitTree(tree));
  // read the formula piece by piece at its top level and join the pieces with pauses (phrasing.mjs)
  return speakPhrased(tree, (nodes) => sre.toSpeech(`<math>${nodes.map((n) => visitor.visitTree(n)).join("")}</math>`),
    (whole) => sre.toSpeech(visitor.visitTree(whole)));
}

function speakLine(line) {
  return line.replace(/\$([^$]+)\$/g, (_, latex) => {
    try {
      return speakFormula(latex);
    } catch (e) {
      if (process.env.DEBUG) console.error(e); return latex; // fall back to the raw LaTeX if it cannot be parsed
    }
  });
}

if (args.includes("--serve")) {
  // long-running mode for streaming: one line in, one line out, so the caller pays Node's start-up once
  const rl = require("readline").createInterface({ input: process.stdin });
  rl.on("line", (line) => process.stdout.write(speakLine(line) + "\n"));
} else {
  const input = require("fs").readFileSync(0, "utf8");
  for (const line of input.split("\n")) {
    if (!line.trim()) continue;
    console.log(speakLine(line));
  }
}
