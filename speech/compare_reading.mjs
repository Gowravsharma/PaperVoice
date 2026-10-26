// compare_reading.mjs - show ClearSpeak's literal reading next to the readable.mjs reading.
// Usage: node compare_reading.mjs 'v^{h}(t,x)' '\mathbb{R}^{d}' ...
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
await sre.setupEngine({ locale: "en", domain: "clearspeak", modality: "speech" }); await sre.engineReady();
for (const t of process.argv.slice(2)) {
  const m = doc.convert(t, { display: false, end: STATE.CONVERT });
  const before = sre.toSpeech(v.visitTree(m));
  const after = sre.toSpeech(v.visitTree(makeReadable(m, doc.mmlFactory)));
  console.log(`${t}\n   before: ${before}\n   after : ${after}`);
}
