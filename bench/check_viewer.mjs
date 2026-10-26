// check_viewer.mjs - every sentence of a paper through the real viewer in headless Edge: is the right text lit,
// does it fall back to the blue paragraph box, and does a Ctrl+click in it start that sentence.
//
// Usage: node bench/check_viewer.mjs <edge.exe> <profile dir> <viewer url> <pdf> [--pages 1-4] [--show 12]
// Needs a running papervoice-server at <viewer url>. For each sentence in reading order (as the player meets them):
//   lit        every lit word is over its own letters, the band (first to last lit word) is the sentence's, and
//              at least 80 % of its words are lit (the rest: maths the text layer prints differently)
//   partly     nothing over other text, but fewer than 80 % of its words lit
//   wrong      some lit word is over other text, or the band starts or ends on another sentence's word
//   box        no place found: the blue paragraph box is shown instead (text sentences only)
//   overlap    its band shares letters with another sentence's band: lit where another sentence is printed
//              (right letters in the wrong place - e.g. a sentence's last words "et al., 2019)." found inside a
//              later sentence - which the letter checks above cannot see)
//   click ok   Ctrl+click at the middle of its first lit word starts that sentence (the click handler's own
//              code: clickContext -> locateQuery -> /locate)
import { spawn } from "node:child_process";

const [edge, profile, url, pdfPath, ...rest] = process.argv.slice(2);
const opt = (name, dflt) => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : dflt; };
const [lo, hi] = opt("--pages", "1-999").split("-").map(Number);
const show = +opt("--show", 12);
const port = 9341;
const proc = spawn(edge, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, "--window-size=1300,1000", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 40; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(250); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map(), errors = [];
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  if (msg.method === "Runtime.exceptionThrown") errors.push(msg.params.exceptionDetails.exception?.description);
});
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => {
  const res = (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result;
  if (res.exceptionDetails) throw new Error(res.exceptionDetails.exception?.description || "evaluate failed");
  return res.result.value;
};

await send("Runtime.enable");
await send("DOM.enable");
await send("Page.navigate", { url });
await sleep(1500);
const { root } = (await send("DOM.getDocument")).result;
const { nodeId } = (await send("DOM.querySelector", { nodeId: root.nodeId, selector: "#file" })).result;
await send("DOM.setFileInputFiles", { nodeId, files: [pdfPath] });
for (let i = 0; i < 900; i++) {
  await sleep(100);
  if (await evaluate(`!!(window.pdfReader.state.doc && document.querySelectorAll(".textLayer span").length > 20)`)) break;
}
await evaluate(`window.pdfReader.state.hlMode = "sentence"; window.pdfReader.state.follow = false`);

// every page prepared, then the whole reading order
const nPages = await evaluate("window.pdfReader.state.pdf.numPages");
const last = Math.min(hi, nPages);
for (let p = 1; p <= last; p++) await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/page/${p}").then(r => r.json())`);
const first = await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/page/1").then(r => r.json()).then(j => j.blocks[0].sentences[0].id)`);
const items = (await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/sequence?from=${first}&count=5000").then(r => r.json())`))
  .filter((it) => it.page >= lo && it.page <= last && ["text", "caption", "heading"].includes(it.kind));

const counts = { lit: 0, partly: 0, wrong: 0, box: 0, overlap: 0, clickOk: 0, clickTried: 0 };
const bad = [];
for (const it of items) {
  const r = await evaluate(`(async () => {
    const R = window.pdfReader, it = ${JSON.stringify(it)};
    for (const n of [it.page, it.prev_part && it.prev_part.page]) if (n) await R.render(R.state.pages[n - 1]);
    R.highlight(it, false);
    const cur = R.state.cur, words = it.text.split(/\\s+/).filter(Boolean).map(R.norm);
    if (!cur.sentence) return cur.prev ? { unlitHere: true } : { box: true }; // (no box when the first part is lit)
    // every lit word on both pages, as showSentence() lights them: on the previous page the words up to the first
    // one found here (that one can be on both: a word hyphenated across the page break, its start there and its
    // end here - only those two are right as part of the word)
    const firstHere = cur.sentence.words.findIndex(Boolean);
    const thereKs = cur.prev ? words.map((_, k) => k).filter((k) => cur.prev.sentence.words[k] && (firstHere < 0 || k <= firstHere)) : [];
    const lastThere = thereKs.length ? thereKs[thereKs.length - 1] : -1;
    const placements = [ // [k, sentence, page, part allowed: "end" | "start" | null], in reading order
      ...thereKs.map((k) => [k, cur.prev.sentence, cur.prev.p, k === lastThere ? "start" : null]),
      ...words.map((_, k) => k).filter((k) => cur.sentence.words[k]).map((k) => [k, cur.sentence, cur.p, cur.prev && k === firstHere ? "end" : null])];
    const right = new Set(), wrongWords = new Set(), ends = [];
    let firstLit = null;
    for (const [k, src, pg, part] of placements) {
      const w = words[k];
      if (!w) continue;
      const rng = src.words[k];
      const got = R.norm(src.chars.slice(rng[0], rng[1]).map(c => c.ch).join(""));
      const ok = got === w || (got.length >= 2 && (part === "end" ? w.endsWith(got) : part === "start" ? w.startsWith(got) : false));
      ends.push(ok);
      if (ok) {
        right.add(k);
        if (!firstLit && pg.n === it.page && src === cur.sentence) firstLit = src.chars[Math.floor((rng[0] + rng[1]) / 2)];
      } else wrongWords.add(k);
    }
    const total = words.filter(Boolean).length, placed = right.size, mis = wrongWords.size;
    const bandOk = ends.length > 0 && ends[0] && ends[ends.length - 1];
    // the bands showSentence() draws (first to last lit word, on each page) -> who lit each letter
    const lit = window.__lit || (window.__lit = new Map());
    const keyOf = (pg, c) => {
      if (!pg.__idx || pg.__idx.arr !== pg.textDivs) { pg.__idx = new Map(pg.textDivs.map((x, i) => [x, i])); pg.__idx.arr = pg.textDivs; }
      return pg.n + ":" + pg.__idx.get(c.span) + ":" + c.i;
    };
    const bands = [];
    const here = cur.sentence.words.filter(Boolean);
    if (thereKs.length) {
      const there = thereKs.map((k) => cur.prev.sentence.words[k]);
      bands.push([cur.prev.p, cur.prev.sentence.chars.slice(there[0][0], there[there.length - 1][1])]);
    }
    if (here.length) bands.push([cur.p, cur.sentence.chars.slice(here[0][0], here[here.length - 1][1])]);
    let overlapWith = null;
    for (const [pg, cs] of bands) for (const c of cs) {
      const key = keyOf(pg, c), owner = lit.get(key);
      if (owner && owner !== it.id) overlapWith = overlapWith || owner;
      else lit.set(key, it.id);
    }
    let click = null;
    if (firstLit) { // the click handler's own path (clickContext -> locateQuery -> /locate), at the word's middle
      firstLit.span.scrollIntoView({ block: "center" });
      await new Promise(r => requestAnimationFrame(r));
      const range = document.createRange(); range.setStart(firstLit.node, firstLit.i); range.setEnd(firstLit.node, firstLit.i + 1);
      const b = range.getBoundingClientRect();
      const ctx = R.clickContext(R.state.pages[it.page - 1], firstLit.span, b.left + b.width / 2, b.top + b.height / 2);
      click = (await (await fetch("/api/doc/" + R.state.doc + "/locate?" + R.locateQuery(ctx))).json()).sentence;
      if (click !== it.id) click += "  [line " + JSON.stringify(ctx.text) + " at " + ctx.at + "]";
    }
    return { placed, total, mis, bandOk, overlapWith, click };
  })()`);
  let verdict;
  if (r.box) { counts.box++; verdict = "box"; }
  else if (r.unlitHere) { counts.partly++; verdict = "partly (its part on this page unlit)"; }
  else if (r.mis || !r.bandOk) { counts.wrong++; verdict = "wrong"; }
  else if (r.overlapWith) { counts.overlap++; verdict = `overlap (with ${r.overlapWith})`; }
  else if (r.total && r.placed / r.total >= 0.8) { counts.lit++; verdict = "lit"; }
  else { counts.partly++; verdict = "partly"; }
  if (r.click !== undefined && r.click !== null) {
    counts.clickTried++;
    if (r.click === it.id) counts.clickOk++; // (a miss carries the clicked line, for the report)
    else bad.push(`click ${it.id} -> ${r.click}: ${JSON.stringify(it.text.slice(0, 70))}`);
  }
  if (verdict !== "lit") bad.push(`${verdict} ${it.id} (${r.placed ?? 0}/${r.total ?? "?"} words right, ${r.mis ?? 0} over other text): ` +
    JSON.stringify(it.text.slice(0, 70)));
}
console.log(`${items.length} sentences (pages ${lo}-${last}): lit correctly ${counts.lit}, partly lit ${counts.partly}, ` +
  `over other text ${counts.wrong}, over another sentence ${counts.overlap}, blue box ${counts.box}; ` +
  `Ctrl+click starts that sentence ${counts.clickOk}/${counts.clickTried}`);
for (const line of bad.slice(0, show)) console.log("   " + line);
if (bad.length > show) console.log(`   ... ${bad.length - show} more`);
console.log("errors:", errors.length ? errors : "none");
ws.close();
proc.kill();
process.exit(0);
