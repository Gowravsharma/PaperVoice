// sre_compare.mjs - read a test set of formulas under several speech-engine settings, side by side.
// Usage: node sre_compare.mjs [--ssml]     (--ssml also shows the engine's SSML output with pause marks)
import { createRequire } from "module";
import { makeReadable } from "./readable.mjs";
const require = createRequire(import.meta.url);
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const { SerializedMmlVisitor } = require("mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js");
const { STATE } = require("mathjax-full/js/core/MathItem.js");
const sre = require("speech-rule-engine");

RegisterHTMLHandler(liteAdaptor());
const doc = mathjax.document("", { InputJax: new TeX({ packages: AllPackages.filter((p) => p !== "bussproofs") }) });
const v = new SerializedMmlVisitor();
const mml = (tex) => v.visitTree(makeReadable(doc.convert(tex, { display: false, end: STATE.CONVERT }), doc.mmlFactory));

// the plan's categories, with expressions taken from the test papers where possible
export const TESTS = [
  ["subscript", String.raw`x_{i}+y_{j,k}`],
  ["function of a pair", String.raw`\varphi(t,x)`],
  ["power / index", String.raw`\sigma^{2}+v^{h}(t,x)`],
  ["fraction", String.raw`\frac{\partial\varphi}{\partial t}+\frac{1}{2}a`],
  ["nested fraction", String.raw`\frac{x-t}{\tau}\psi\left(\frac{x-t}{\tau}\right)`],
  ["root", String.raw`\sqrt{h}N+\sqrt[3]{x}`],
  ["sum", String.raw`\sum_{i=1}^{10}W_{i}(t)`],
  ["integral", String.raw`\int_{t_i}^{t_{i+1}}F(\cdot,v)\,dt`],
  ["sup / limit", String.raw`\sup_{(t,x)\in Q_T}|\varphi(t,x)|`],
  ["limsup", String.raw`\limsup_{h\to 0}h^{-1}|v^h-v|_{\infty}`],
  ["absolute value / norm", String.raw`|x-x'|+\|R\|_{p}`],
  ["membership / inequality", String.raw`x\in\mathbb{R}^{d},\ 0\le b<a`],
  ["definition", String.raw`Q_T:=(0,T]\times\mathbb{R}^d`],
  ["transpose / trace", String.raw`\mathrm{Tr}[\sigma^{\mathrm{T}}\sigma]`],
  ["inner product", String.raw`\langle x,y\rangle`],
  ["piecewise", String.raw`f(x)=\begin{cases}x & x\ge 0\\ 0 & x<0\end{cases}`],
  ["min / max (wedge, vee)", String.raw`a\wedge b,\ a\vee b,\ a^{+}`],
];

const SETTINGS = [
  ["clearspeak (current)", { domain: "clearspeak", style: "default" }],
  ["clearspeak Paren_Silent", { domain: "clearspeak", style: "Paren_Silent" }],
  ["clearspeak Fraction_Over", { domain: "clearspeak", style: "Fraction_Over" }],
  ["clearspeak AbsoluteValue_AbsEnd", { domain: "clearspeak", style: "AbsoluteValue_AbsEnd" }],
  ["mathspeak brief", { domain: "mathspeak", style: "brief" }],
];

const wantSsml = process.argv.includes("--ssml");
const results = {};
for (const [name, opts] of SETTINGS) {
  await sre.setupEngine({ locale: "en", modality: "speech", markup: "none", ...opts });
  await sre.engineReady();
  results[name] = TESTS.map(([, tex]) => { try { return sre.toSpeech(mml(tex)); } catch (e) { return "ERROR " + e.message; } });
}
let ssml = [];
if (wantSsml) {
  await sre.setupEngine({ locale: "en", modality: "speech", domain: "clearspeak", style: "default", markup: "ssml" });
  await sre.engineReady();
  ssml = TESTS.map(([, tex]) => { try { return sre.toSpeech(mml(tex)); } catch (e) { return "ERROR " + e.message; } });
}
TESTS.forEach(([label, tex], i) => {
  console.log(`\n### ${label}:  ${tex}`);
  for (const [name] of SETTINGS) console.log(`  ${name.padEnd(32)} ${results[name][i]}`);
  if (wantSsml) console.log(`  ${"SSML (clearspeak)".padEnd(32)} ${ssml[i]}`);
});
