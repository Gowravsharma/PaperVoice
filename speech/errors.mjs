// errors.mjs - print the MathJax TeX error for each input line that fails to parse (debugging aid).
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const { STATE } = require("mathjax-full/js/core/MathItem.js");
RegisterHTMLHandler(liteAdaptor());
const doc = mathjax.document("", { InputJax: new TeX({ packages: AllPackages.filter((p) => p !== "bussproofs") }) });
const lines = require("fs").readFileSync(0, "utf8").split("\n").filter((l) => l.trim());
lines.forEach((l, i) => {
  const m = doc.convert(l, { display: false, end: STATE.CONVERT });
  let err = null;
  m.walkTree((n) => { if (!err && n.kind === "merror") err = n.attributes.get("data-mjx-error") || "merror"; });
  if (err) console.log(`${i}\t${err}`);
});
