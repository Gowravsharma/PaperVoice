// check_playback.mjs - real Ctrl+clicks and real playback in headless Edge: does a click read the clicked sentence,
// and while the voice reads on, is what is shaded on screen always the sentence being spoken?
//
// Usage: node bench/check_playback.mjs <edge.exe> <profile dir> <viewer url> <pdf> [--clicks 8] [--listen 9]
// Needs a running papervoice-server at <viewer url>. Picks sentences spread over the paper (and every sentence
// that goes on from one page to the next, clicked on its second part), Ctrl+clicks the middle of one of their
// words with the mouse (Input.dispatchMouseEvent, as a person does), then listens for --listen seconds, four
// samples a second. Each sample reads the screen, not the viewer's state: the letters of the text layer under
// the shaded bands must be the spoken sentence's (at least 90 % of their words of 4+ letters), something must be
// shaded while a text sentence is spoken, and the blue paragraph box must not appear for one.
import { spawn } from "node:child_process";

const [edge, profile, url, pdfPath, ...rest] = process.argv.slice(2);
const opt = (name, dflt) => { const i = rest.indexOf(name); return i >= 0 ? rest[i + 1] : dflt; };
const nClicks = +opt("--clicks", 8), listen = +opt("--listen", 9);
const port = 9343;
const proc = spawn(edge, ["--headless=new", "--disable-gpu", "--autoplay-policy=no-user-gesture-required",
  `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "--window-size=1300,1000", "about:blank"], { stdio: "ignore" });
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
const mouse = (type, x, y) => send("Input.dispatchMouseEvent", { type, x, y, button: "left", clickCount: 1, modifiers: 2 });

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
await evaluate(`window.pdfReader.state.hlMode = "sentence"`);

// the reading order of the first pages; the sentences to click
const nPages = Math.min(6, await evaluate("window.pdfReader.state.pdf.numPages"));
for (let p = 1; p <= nPages; p++) await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/page/${p}").then(r => r.json())`);
const first = await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/page/1").then(r => r.json()).then(j => j.blocks[0].sentences[0].id)`);
const items = (await evaluate(`fetch("/api/doc/" + window.pdfReader.state.doc + "/sequence?from=${first}&count=3000").then(r => r.json())`))
  .filter((it) => it.page <= nPages && it.kind === "text" && it.text.split(" ").length >= 6);
const picks = [];
for (let k = 0; k < nClicks; k++) picks.push(items[Math.floor(((k + 0.5) * items.length) / nClicks)]);
for (const it of items) if (it.prev_part && !picks.includes(it)) picks.push(it); // every page-break sentence

// a word of the sentence to click: one of its own lit words on its page, after the first few (mid-sentence)
async function target(it) {
  return evaluate(`(async () => {
    const R = window.pdfReader, it = ${JSON.stringify(it)};
    for (const n of [it.page, it.prev_part && it.prev_part.page]) if (n) await R.render(R.state.pages[n - 1]);
    R.highlight(it, false);
    const s = R.state.cur.sentence;
    R.player.stop(); // (highlight() above was only to find the word; nothing is playing yet)
    if (!s) return null;
    const ks = s.words.map((r, k) => r && r[1] - r[0] >= 3 ? k : -1).filter((k) => k >= 0);
    if (!ks.length) return null;
    const k = ks[Math.min(ks.length - 1, Math.floor(ks.length / 2))], r = s.words[k];
    const c = s.chars[Math.floor((r[0] + r[1]) / 2)];
    c.span.scrollIntoView({ block: "center" });
    await new Promise((res) => setTimeout(res, 400));
    const range = document.createRange(); range.setStart(c.node, c.i); range.setEnd(c.node, c.i + 1);
    const b = range.getBoundingClientRect();
    return { x: b.left + b.width / 2, y: b.top + b.height / 2, word: it.text.split(" ")[k] };
  })()`);
}

// what is shaded on screen now: the text-layer letters under the bands, and the sentence being spoken
const SAMPLE = `(() => {
  const R = window.pdfReader, pc = R.player.current;
  const bands = [...document.querySelectorAll(".marks .word.sentence")].map((b) => b.getBoundingClientRect());
  let text = "";
  for (const p of R.state.pages) {
    if (!p.rendered || !p.textDivs) continue;
    for (const span of p.textDivs) {
      const sr = span.getBoundingClientRect();
      if (!bands.some((b) => sr.right > b.left && sr.left < b.right && sr.bottom > b.top && sr.top < b.bottom)) continue;
      const node = span.firstChild;
      if (!node || node.nodeType !== 3) continue;
      for (let i = 0; i < node.data.length; i++) {
        const range = document.createRange(); range.setStart(node, i); range.setEnd(node, i + 1);
        const r = range.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
        if (bands.some((b) => cx >= b.left && cx <= b.right && cy >= b.top && cy <= b.bottom)) text += node.data[i];
        else text += " ";
      }
      text += " ";
    }
  }
  return { id: pc && pc.id, kind: pc && pc.kind, said: pc && pc.text, playing: !R.player.audio.paused,
           bands: bands.length, box: !!document.querySelector(".marks .mark"), shaded: text };
})()`;
const letters = (s) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");

let clickOk = 0, samples = 0, wrongShade = 0, unshaded = 0, boxes = 0;
const notes = [];
for (const it of picks) {
  const t = await target(it);
  if (!t) { notes.push(`no word to click in ${it.id}`); continue; }
  await mouse("mousePressed", t.x, t.y);
  await mouse("mouseReleased", t.x, t.y);
  let cur = null;
  for (let i = 0; i < 200 && !cur; i++) {
    await sleep(100);
    cur = await evaluate("(() => { const P = window.pdfReader.player; return P.current && !P.audio.paused ? P.current.id : null; })()");
  }
  if (cur === it.id) clickOk++;
  else notes.push(`Ctrl+click on "${t.word}" in ${it.id} read ${cur}: ${JSON.stringify(it.text.slice(0, 60))}`);
  const seen = new Set();
  for (let n = 0; n < listen * 4; n++) {
    await sleep(250);
    const s = await evaluate(SAMPLE);
    if (!s.id || !s.playing || s.kind !== "text") continue;
    samples++;
    seen.add(s.id);
    if (s.box) { boxes++; notes.push(`blue box while reading ${s.id}`); }
    if (!s.bands) { unshaded++; notes.push(`nothing shaded while reading ${s.id}`); continue; }
    const said = letters(s.said);
    const words = s.shaded.split(/\s+/).map(letters).filter((w) => w.length >= 4);
    const inIt = words.filter((w) => said.includes(w)).length;
    if (words.length && inIt / words.length < 0.9) {
      wrongShade++;
      notes.push(`while reading ${s.id} the shading shows ${JSON.stringify(s.shaded.replace(/\s+/g, " ").trim().slice(0, 70))}`);
    }
  }
  await evaluate("window.pdfReader.player.stop()");
}
console.log(`${picks.length} Ctrl+clicks: read the clicked sentence ${clickOk}/${picks.length}; ${samples} samples while ` +
  `reading: shading on another sentence ${wrongShade}, nothing shaded ${unshaded}, blue box ${boxes}`);
for (const line of notes.slice(0, 25)) console.log("   " + line);
console.log("errors:", errors.length ? errors : "none");
ws.close();
proc.kill();
process.exit(0);
