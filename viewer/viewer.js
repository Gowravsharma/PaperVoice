// viewer.js - read-along PDF viewer. PDF.js draws the pages; the local server (papervoice-server) supplies the
// index (blocks, positions, sentences) and the voice. The spoken sentence is highlighted; a Ctrl+click anywhere
// starts reading from that point.
import * as pdfjsLib from "/pdfjs/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.mjs";

const $ = (id) => document.getElementById(id);
const pagesEl = $("pages");
const state = { doc: null, pdf: null, pages: [], speed: 0.8, follow: true, lit: [], mark: null, zoom: 1 };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function status(text) {
  $("status").textContent = text;
}

async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).error || msg; } catch {}
    throw new Error(msg);
  }
  return r;
}

// letters and digits only, lower case: the printed sentence and the PDF.js text layer agree on these even when
// spacing, hyphens or ligatures differ
const norm = (s) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");

// ------------------------------------------------------------------ opening a document
async function waitForServer() {
  for (;;) {
    try {
      const h = await (await fetch("/api/health")).json();
      if (h.ready) return;
      status("Loading the reading models…");
    } catch {
      status("Waiting for papervoice-server…");
    }
    await sleep(1000);
  }
}

async function openBytes(buf, name) {
  // show the PDF at once; the reading engine may still be loading its models (a cold start takes 15-60 s),
  // so connect to it in the background - a click made meanwhile starts reading as soon as it is ready
  player.stop();
  state.doc = null;
  showName(name);
  state.pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buf.slice(0)) }).promise;
  await buildPages();
  state.docReady = (async () => {
    await waitForServer();
    const info = await (await api("/api/open", {
      method: "POST", headers: { "Content-Type": "application/pdf" }, body: buf,
    })).json();
    state.doc = info.doc;
    status(`${state.pdf.numPages} pages · Ctrl+click in the text to start`);
    return info.doc;
  })();
  state.docReady.catch((err) => status(`Error: ${err.message}`));
}

function showName(name) {
  $("name").textContent = name;
  document.title = `${name} – PaperVoice`;
}

async function openPath(path) {
  status("Opening…");
  const info = await (await api("/api/open", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }),
  })).json();
  const buf = await (await api(`/api/doc/${info.doc}/pdf`)).arrayBuffer();
  await load({ ...info, name: path.split(/[\\/]/).pop() }, buf);
}

async function load(info, buf) {
  player.stop();
  state.doc = info.doc;
  state.docReady = Promise.resolve(info.doc);
  showName(info.name);
  // PDF.js hands the buffer to its worker (which detaches it), so give it a copy
  state.pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buf.slice(0)) }).promise;
  await buildPages();
  status(`${state.pdf.numPages} pages · Ctrl+click in the text to start`);
}

// ------------------------------------------------------------------ pages (rendered when near the view)
let observer = null;

async function buildPages() {
  pagesEl.innerHTML = "";
  state.pages = [];
  const first = await state.pdf.getPage(1);
  const base = first.getViewport({ scale: 1 });
  // zoom 1 = the page fills the width of the window; pages are drawn (not stretched) at this size, so they stay
  // sharp at any zoom, as in the browser's own PDF viewer
  const fit = Math.max(0.3, (pagesEl.clientWidth - 48) / base.width);
  const scale = Math.min(8, fit * state.zoom);
  observer?.disconnect();
  observer = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) render(state.pages[+e.target.dataset.page - 1]);
  }, { root: pagesEl, rootMargin: "900px 0px" });
  for (let n = 1; n <= state.pdf.numPages; n++) {
    const div = document.createElement("div");
    div.className = "page";
    div.dataset.page = n;
    div.style.width = `${base.width * scale}px`;
    div.style.height = `${base.height * scale}px`;
    div.style.setProperty("--scale-factor", scale);
    div.innerHTML = `<span class="num">Page ${n}</span>`;
    pagesEl.append(div);
    state.pages.push({ n, div, scale, rendering: null, rendered: false, textDivs: [] });
    observer.observe(div);
  }
}

function render(p) {
  if (!p || p.rendering) return p?.rendering;
  p.rendering = (async () => {
    const page = await state.pdf.getPage(p.n);
    const viewport = page.getViewport({ scale: p.scale });
    p.viewport = viewport;
    p.view = page.view; // [x0, y0, x1, y1] in PDF points
    p.div.style.width = `${viewport.width}px`;
    p.div.style.height = `${viewport.height}px`;
    const dpr = window.devicePixelRatio || 1;
    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(viewport.width * dpr);
    canvas.height = Math.floor(viewport.height * dpr);
    p.div.append(canvas);
    await page.render({
      canvasContext: canvas.getContext("2d"), viewport,
      transform: dpr !== 1 ? [dpr, 0, 0, dpr, 0, 0] : null,
    }).promise;
    const tl = document.createElement("div");
    tl.className = "textLayer";
    p.div.append(tl);
    const layer = new pdfjsLib.TextLayer({ textContentSource: page.streamTextContent(), container: tl, viewport });
    await layer.render();
    p.textDivs = layer.textDivs || [...tl.querySelectorAll("span:not(.markedContent)")];
    p.marks = document.createElement("div");
    p.marks.className = "marks";
    p.div.append(p.marks);
    p.rendered = true;
    if (player.current?.page === p.n || player.current?.prev_part?.page === p.n) highlight(player.current, false);
  })();
  return p.rendering;
}

// index coordinates: PDF points, origin top-left of the page  <->  pixels inside the page div
function toView(p, bbox) {
  if (!bbox || !p.viewport) return null;
  const [l, t, r, b] = bbox;
  const [x0, , , y1] = p.view;
  const [a, c, d, e] = p.viewport.convertToViewportRectangle([l + x0, y1 - t, r + x0, y1 - b]);
  return { left: Math.min(a, d), top: Math.min(c, e), width: Math.abs(d - a), height: Math.abs(e - c) };
}

function toPdf(p, x, y) {
  const [px, py] = p.viewport.convertToPdfPoint(x, y);
  return { x: px - p.view[0], y: p.view[3] - py };
}

// ------------------------------------------------------------------ highlighting: one word (or one symbol)
// state.cur describes the sentence being read: where its characters are in the text layer, where each printed
// word starts and ends, and (once the audio's timing arrives) which word or symbol to light at each moment.

function clearHighlight() {
  for (const el of state.lit) el.remove();
  state.lit = [];
  state.mark?.remove();
  state.mark = null;
}

function charsInBox(p, box) {
  // every visible character of the text layer inside the block, in reading (content) order
  const pageRect = p.div.getBoundingClientRect();
  const out = [];
  for (const span of p.textDivs) {
    const r = span.getBoundingClientRect();
    const cx = r.left + r.width / 2 - pageRect.left, cy = r.top + r.height / 2 - pageRect.top;
    if (cx < box.left - 3 || cx > box.left + box.width + 3 || cy < box.top - 3 || cy > box.top + box.height + 3) continue;
    const node = span.firstChild;
    if (!node || node.nodeType !== Node.TEXT_NODE) continue;
    for (let i = 0; i < node.data.length; i++) if (node.data[i].trim()) out.push({ span, node, i, ch: node.data[i] });
  }
  return out;
}

function charsInBoxes(p, boxes) {
  // the characters of all of a block's boxes in reading order (a paragraph that goes on in the next column has
  // two); a character inside two overlapping boxes is taken once
  const seen = new Map(), out = []; // text node -> offsets already taken
  for (const b of boxes) {
    const view = toView(p, b);
    if (!view) continue;
    for (const c of charsInBox(p, view)) {
      let taken = seen.get(c.node);
      if (!taken) seen.set(c.node, (taken = new Set()));
      if (taken.has(c.i)) continue;
      taken.add(c.i);
      out.push(c);
    }
  }
  return out;
}

function locateSentence(chars, text, from = 0, atStart = false) {
  // the sentence's printed words inside the block's characters -> {chars, words: [[a, b) or null], end}
  // Each word is looked for near where the previous one ended, so a symbol that only one of the two texts has
  // (Docling's text vs the PDF.js text layer: maths, superscripts, footnote marks) cannot shift the rest of the
  // sentence; the search starts after the previous sentence of the same block (from), so a repeated phrase cannot
  // pull the highlight elsewhere. Words that are not in this block (a sentence begun on the previous page) stay
  // unlit. Checked on every sentence of a paper with bench/check_highlight.py.
  const letters = chars.filter((c) => /[\p{L}\p{N}]/u.test(c.ch));
  let flat = "";
  const owner = []; // flat index -> letters index ("ﬁ" is one glyph, two letters after NFKC)
  letters.forEach((c, k) => {
    for (const ch of c.ch.normalize("NFKC").toLowerCase()) {
      if (/[\p{L}\p{N}]/u.test(ch)) { flat += ch; owner.push(k); }
    }
  });
  const ws = text.split(/\s+/).filter(Boolean).map(norm);
  if (!flat || !ws.length) return null;
  // the start: where the sentence's words follow each other on the page. A candidate (three words in a row, then
  // two, then one long word) is accepted only if the next words of the sentence follow it too - a phrase such as
  // "et al. 2019" or "compared to the" can occur again later in the block. A sentence begun on the previous page
  // has only its end here, so its first words are skipped until a verified run is found.
  const follows = (k, at, upto) => { // how many of words k.. appear one after another from letter `at`
    let ok = 0, skipped = 0;
    for (let i = k; i < Math.min(ws.length, k + upto); i++) {
      if (!ws[i]) continue;
      const j = flat.indexOf(ws[i], at);
      if (j < 0 || j > at + 3) {
        // a one- or two-letter word is often a maths letter or a sub/superscript, which the text layer can hold
        // in another order ("||·||c" in DoRA): up to two of them may be elsewhere, the longer words must follow
        if (ws[i].length <= 2 && skipped++ < 2) { ok++; continue; }
        break;
      }
      ok++;
      at = j + ws[i].length;
    }
    return ok;
  };
  let pos = -1, first = 0, best = null, split = 0;
  if (atStart) {
    // the rest of a sentence begun on the previous page: it is the start of this block (that is what joining
    // the two parts means), so it is only looked for there - its last words, searched anywhere in the block,
    // can be a phrase of a later sentence ("et al., 2019)." in FastSpeech 2 p2). The first word here is the
    // first one at the block's first letters that the next words follow (checked as below: up to 8 of them).
    // A word hyphenated across the page break ("mag-" / "nitude") begins this page with its end only.
    for (let k = 0; k < ws.length && pos < 0; k++) {
      if (!ws[k]) continue;
      const j = flat.indexOf(ws[k]);
      if (j >= 0 && j <= 3 && follows(k, j, 8) >= ws.slice(k, k + 8).filter(Boolean).length) { pos = j; first = k; break; }
      for (let n = ws[k].length - 1; n >= 2; n--) { // the word's last n letters at the start
        if (!flat.startsWith(ws[k].slice(-n))) continue;
        if (follows(k + 1, n, 8) >= ws.slice(k + 1, k + 9).filter(Boolean).length) { pos = 0; first = k; split = n; }
        break;
      }
    }
    if (pos < 0) return null; // not placed: unlit here rather than lit over other words
  }
  // a sentence of only tiny words ("0 C, 5°, -5°": 4 words, 4 letters) has no run of 4+ letters in 3 words or
  // fewer: it is looked for whole
  const tiny = ws.length > 3 && ws.join("").length < 8;
  search: for (const n of pos >= 0 ? [] : tiny ? [ws.length, 3, 2, 1] : [3, 2, 1]) {
    for (let k = 0; k + n <= ws.length; k++) {
      const key = ws.slice(k, k + n).join("");
      if (key.length < (n === 1 ? 5 : 4) && !(k === 0 && n === ws.length)) continue; // short: only all of it
      // the next 8 words that have letters (follows() skips symbols such as "||" or "·": counting them here
      // would make a sentence with a symbol near its start impossible to verify at its first words)
      const need = ws.slice(k, k + 8).filter(Boolean).length;
      for (const start of [from, 0]) {
        for (let j = flat.indexOf(key, start); j >= 0; j = flat.indexOf(key, j + 1)) {
          const ok = follows(k, j, 8);
          if (ok >= need) { pos = j; first = k; break search; }
          if (!best || ok > best.ok) best = { ok, j, k };
        }
        if (start === 0) break;
      }
    }
  }
  if (pos < 0 && best && best.ok >= 3) { pos = best.j; first = best.k; }
  if (pos < 0) return null;
  const words = ws.map(() => null);
  if (split) { // its end, at the block's start
    words[first] = [owner[0], owner[split - 1] + 1];
    pos = split;
  }
  for (let k = first + (split ? 1 : 0); k < ws.length; k++) {
    const w = ws[k];
    if (!w) continue;
    const rest = flat.length - pos;
    if (rest >= 2 && rest < w.length && w.startsWith(flat.slice(pos))) { // the block (page) ends inside this word:
      words[k] = [owner[pos], owner[flat.length - 1] + 1]; // "mag-" of "magnitude" before a page break
      pos = flat.length;
      continue;
    }
    const j = flat.indexOf(w, pos);
    // not found close by (printed differently, or not on this page): left unlit, and the next words are still
    // looked for from here - lighting the next letters instead would push every later word off its place
    if (j < 0 || j > pos + 40) continue;
    words[k] = [owner[j], owner[j + w.length - 1] + 1];
    pos = j + w.length;
  }
  return { chars: letters, words, end: pos };
}

// spoken formula word -> printed symbols it can stand for (literal reading, as in the speech rules)
const GREEK = { alpha: "α", beta: "β", gamma: "γΓ", delta: "δΔ∆", epsilon: "εϵ", zeta: "ζ", eta: "η", theta: "θϑΘ",
  iota: "ι", kappa: "κ", lambda: "λΛ", lamda: "λΛ", mu: "μµ", nu: "ν", xi: "ξΞ", pi: "πΠ", rho: "ρϱ", sigma: "σΣ",
  tau: "τ", upsilon: "υ", phi: "φϕΦ", chi: "χ", psi: "ψΨ", omega: "ωΩ" };
const SPOKEN_SYMBOL = { ...GREEK, equals: "=", plus: "+", minus: "−-", negative: "−-", times: "×·∗", less: "<≤≪",
  greater: ">≥≫", infinity: "∞", integral: "∫", sum: "∑Σ", product: "∏", partial: "∂", nabla: "∇", prime: "′'",
  comma: ",", period: ".", colon: ":", paren: "()", bracket: "[]", brace: "{}", bar: "|‖", vertical: "|",
  member: "∈", arrow: "→⟶↦⇒", hat: "ˆ^", dot: "·.", root: "√", tilde: "˜~", zero: "0", one: "1", two: "2",
  three: "3", four: "4", five: "5", six: "6", seven: "7", eight: "8", nine: "9", ten: "10", half: "2",
  squared: "2", cubed: "3", union: "∪", intersection: "∩", subset: "⊂⊆", approximately: "≈", not: "≠",
  absolute: "|∣" };
// spoken as a word and printed as letters: light the first letter ("sup" -> the s of "sup")
const OPERATOR_NAMES = new Set(["sup", "inf", "max", "min", "lim", "limsup", "liminf", "log", "ln", "exp", "sin",
  "cos", "tan", "det", "dim", "ker"]);

function symbolsFor(word) {
  const w = word.replace(/[^\p{L}\p{N}]/gu, "");
  if (/^[A-Za-z]$/.test(w)) return w; // a variable: x, F, T
  if (OPERATOR_NAMES.has(w.toLowerCase())) return w[0];
  return SPOKEN_SYMBOL[w.toLowerCase()] || "";
}

function alignFormula(tokens, glyphs) {
  // best in-order matching of spoken words to printed symbols (longest common subsequence); each spoken word
  // gets the index of its symbol, or the previous word's symbol if it has none ("sub", "the fraction with")
  const n = tokens.length, m = glyphs.length;
  const sets = tokens.map(symbolsFor);
  const dp = Array.from({ length: n + 1 }, () => new Int16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      const hit = sets[i] && sets[i].includes(glyphs[j].ch) ? 1 + dp[i + 1][j + 1] : 0;
      dp[i][j] = Math.max(hit, dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out = new Array(n).fill(null);
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (sets[i] && sets[i].includes(glyphs[j].ch) && dp[i][j] === 1 + dp[i + 1][j + 1]) { out[i++] = j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  for (let k = 1; k < n; k++) if (out[k] === null) out[k] = out[k - 1];
  return out;
}

function drawSpan(p, chars) {
  // one band per text run over these characters, spaces and punctuation between them included (a whole sentence)
  const runs = [];
  for (const c of chars) {
    const last = runs[runs.length - 1];
    if (last && last.node === c.node) last.end = c.i + 1;
    else runs.push({ node: c.node, start: c.i, end: c.i + 1 });
  }
  return drawChars(p, runs.flatMap((r) => [{ node: r.node, i: r.start }, ...Array.from(
    { length: r.end - r.start - 1 }, (_, k) => ({ node: r.node, i: r.start + k + 1 }))]));
}

function drawChars(p, chars) {
  // a highlight box per line over these characters (a word can wrap); returns the boxes
  const pageRect = p.div.getBoundingClientRect();
  const boxes = [];
  for (let k = 0; k < chars.length;) {
    let e = k;
    while (e + 1 < chars.length && chars[e + 1].node === chars[k].node && chars[e + 1].i === chars[e].i + 1) e++;
    const range = document.createRange();
    range.setStart(chars[k].node, chars[k].i);
    range.setEnd(chars[e].node, chars[e].i + 1);
    for (const r of range.getClientRects()) {
      const div = document.createElement("div");
      div.className = "word";
      Object.assign(div.style, { left: `${r.left - pageRect.left - 1}px`, top: `${r.top - pageRect.top - 1}px`,
        width: `${r.width + 2}px`, height: `${r.height + 2}px` });
      p.marks.append(div);
      boxes.push(div);
    }
    k = e + 1;
  }
  return boxes;
}

function boxMark(p, box) {
  const mark = document.createElement("div");
  mark.className = "mark";
  Object.assign(mark.style, {
    left: `${box.left - 4}px`, top: `${box.top - 2}px`, width: `${box.width + 8}px`, height: `${box.height + 4}px`,
  });
  p.marks.append(mark);
  return mark;
}

function highlight(item, scroll = true) {
  // called when a sentence starts: find it on the page; the word-by-word part runs in tick()
  clearHighlight();
  const p = state.pages[item.page - 1];
  const timing = state.cur?.item === item ? state.cur.timing : null; // keep it when the page re-renders
  state.cur = { item, p, shown: undefined, timing };
  if (!p) return;
  // a sentence that starts on the previous page: its first words are there (item.prev_part), the rest here
  const pp = item.prev_part ? state.pages[item.prev_part.page - 1] : null;
  for (const q of [pp, p]) {
    if (q && !q.rendered) {
      if (scroll && state.follow) (pp || p).div.scrollIntoView({ block: "center", behavior: "auto" });
      render(q);
      return; // render() calls highlight() again when the page is ready
    }
  }
  if (pp && item.text) {
    const pchars = charsInBoxes(pp, item.prev_part.boxes?.length ? item.prev_part.boxes : [item.prev_part.bbox]);
    const sentence = pchars.length ? locateSentence(pchars, item.text) : null;
    if (sentence) state.cur.prev = { p: pp, sentence };
  }
  const box = toView(p, item.bbox); // the first box: where to scroll, and the light box for tables and scans
  if (!box) return;
  const chars = charsInBoxes(p, item.boxes?.length ? item.boxes : [item.bbox]);
  if (item.text && ["text", "caption", "heading"].includes(item.kind)) {
    // continue after the previous sentence when it was in the same block (reading on), else from the top
    const c = state.blockCursor;
    const from = c && c.page === item.page && c.block === item.block ? c.end : 0;
    state.cur.sentence = locateSentence(chars, item.text, from, !!item.prev_part);
    if (state.cur.sentence) state.blockCursor = { page: item.page, block: item.block, end: state.cur.sentence.end };
  }
  else if (item.kind === "formula" && chars.length) state.cur.glyphs = chars;
  // tables, scans: a light box (not for a sentence whose first part is lit on the previous page)
  if (!state.cur.sentence && !state.cur.glyphs && !state.cur.prev) state.mark = boxMark(p, box);
  if (scroll && state.follow) {
    if (state.cur.prev) keepInView(null, toView(pp, item.prev_part.bbox), pp); // the voice starts there
    else keepInView(state.mark || p.marks, box, p);
  }
  if (state.cur.timing) prepareTiming();
  if (state.hlMode === "sentence") showSentence();
  else tick(true);
}

function wordPlace(cur, k) {
  // {p, chars} of printed word k: on this page if it was found here, else on the previous page
  const here = cur.sentence?.words[k];
  if (here) return { p: cur.p, chars: cur.sentence.chars.slice(here[0], here[1]) };
  const there = cur.prev?.sentence.words[k];
  if (there) return { p: cur.prev.p, chars: cur.prev.sentence.chars.slice(there[0], there[1]) };
  return null;
}

function showSentence() {
  // the whole sentence (or formula) being read, lit at once: no word timing involved
  const cur = state.cur;
  for (const el of state.lit) el.remove();
  state.lit = [];
  if (!cur?.p?.rendered) return;
  const parts = []; // [page, chars] - one band per page the sentence is on
  if (cur.sentence || cur.prev) { // (only the previous page's part when this page's could not be placed)
    const words = cur.sentence ? cur.sentence.words : [];
    const ranges = words.filter(Boolean);
    if (cur.prev) { // the words before the first one found here are on the previous page; that first one can
      // be on both (a word hyphenated across the page break: "mag-" there, "nitude" here)
      const firstHere = words.findIndex(Boolean);
      const there = cur.prev.sentence.words.filter((r, k) => r && (firstHere < 0 || k <= firstHere));
      if (there.length) parts.push([cur.prev.p, cur.prev.sentence.chars.slice(there[0][0], there[there.length - 1][1])]);
    }
    if (ranges.length) parts.push([cur.p, cur.sentence.chars.slice(ranges[0][0], ranges[ranges.length - 1][1])]);
  } else if (cur.glyphs) parts.push([cur.p, cur.glyphs]);
  for (const [page, chars] of parts) {
    if (!chars.length) continue;
    const boxes = drawSpan(page, chars);
    for (const el of boxes) el.classList.add("sentence");
    state.lit.push(...boxes);
  }
  if (state.follow && state.lit[0]) keepInView(state.lit[0]); // where the sentence starts
}

function prepareTiming() {
  const cur = state.cur;
  if (cur.glyphs && cur.timing) cur.glyphOf = alignFormula(cur.timing.words.map((w) => w.w), cur.glyphs);
}

function tick(force = false) {
  // which word is being spoken now -> light it (only when it changes); word mode only
  const cur = state.cur;
  if (state.hlMode !== "word" || !cur || !cur.timing || !cur.p?.rendered) return;
  const t = player.audio.currentTime * (cur.timing.duration / (player.audio.duration || cur.timing.duration));
  const words = cur.timing.words;
  let i = cur.wi ?? 0;
  if (words[i] && t < words[i].start) i = 0;
  while (i + 1 < words.length && t >= words[i + 1].start) i++;
  cur.wi = i;
  let chars = null, key = null, page = cur.p;
  if ((cur.sentence || cur.prev) && words[i]?.printed != null) {
    key = words[i].printed;
    const place = wordPlace(cur, key);
    if (place) ({ p: page, chars } = place);
  } else if (cur.glyphs && cur.glyphOf && cur.glyphOf[i] != null) {
    key = cur.glyphOf[i];
    chars = [cur.glyphs[key]];
  }
  if (key === cur.shown && !force) return;
  cur.shown = key;
  for (const el of state.lit) el.remove();
  state.lit = chars && chars.length && page.rendered ? drawChars(page, chars) : [];
  if (state.follow && state.lit[0]) keepInView(state.lit[0]);
}

function keepInView(el, box, p) {
  let r = el?.getBoundingClientRect?.();
  if (box && p) {
    const pr = p.div.getBoundingClientRect();
    r = { top: pr.top + box.top, bottom: pr.top + box.top + Math.min(box.height, 120) };
  }
  if (!r) return;
  const view = pagesEl.getBoundingClientRect();
  if (r.top < view.top + 60 || r.bottom > view.bottom - 60) {
    pagesEl.scrollBy({ top: r.top - view.top - view.height * 0.3, behavior: reducedMotion ? "auto" : "smooth" });
  }
}

function loop() {
  if (player.playing) tick();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// ------------------------------------------------------------------ player
function locateQuery({ page, x, y, text = "", at = null, before = "", after = "" }) {
  return `page=${page}&x=${x.toFixed(1)}&y=${y.toFixed(1)}&text=${encodeURIComponent(text)}` +
    (at !== null ? `&at=${at}` : "") +
    (before || after ? `&before=${encodeURIComponent(before)}&after=${encodeURIComponent(after)}` : "");
}

async function fetchSequence(from, count = 25) {
  return (await api(`/api/doc/${state.doc}/sequence?from=${encodeURIComponent(from)}&count=${count}`)).json();
}

const player = {
  audio: new Audio(),
  queue: [], idx: 0, gen: 0, playing: false, current: null, more: null, urls: new Map(), log: [], lastEnded: null,

  async startAt({ page, x, y, text = "", at = null, before = "", after = "" }) {
    const g = ++this.gen;
    this.audio.pause();
    if (!state.doc) { // the reading engine is still starting: remember the click
      status("The reading engine is still loading - reading starts as soon as it is ready…");
      await state.docReady;
      if (g !== this.gen) return;
    }
    status(`Finding the sentence on page ${page}…`);
    const q = locateQuery({ page, x, y, text, at, before, after });
    const { sentence } = await (await api(`/api/doc/${state.doc}/locate?${q}`)).json();
    if (g !== this.gen) return;
    if (!sentence) return status("Nothing to read after this point");
    await this.startFrom(sentence, g);
  },

  async startFrom(sid, g = ++this.gen) {
    const seq = await fetchSequence(sid);
    if (g !== this.gen) return;
    this.queue = seq;
    this.idx = 0;
    this.playing = true;
    this.play(g);
  },

  async play(g) {
    const item = this.queue[this.idx];
    if (!item) return this.finish();
    this.current = item;
    highlight(item);
    setPlaying(true);
    status(`Page ${item.page} of ${state.pdf.numPages}`);
    const asked = performance.now();
    let clip = null;
    for (let attempt = 0; attempt < 4 && !clip; attempt++) { // a busy server may drop a request: try again
      try {
        clip = await this.url(item.id);
      } catch (err) {
        if (g !== this.gen) return;
        status(`Waiting for the reading engine… (${err.message})`);
        await sleep(1000 * (attempt + 1));
      }
    }
    if (g !== this.gen || !this.playing) return;
    if (!clip) return status("The reading engine is not answering - is papervoice-server running?");
    const { url, timing } = clip;
    // timeline for diagnosing gaps (window.pdfReader.player.log): when audio was asked for, ready, started
    this.log.push({ id: item.id, page: item.page, asked, ready: performance.now(), prevEnded: this.lastEnded });
    if (this.log.length > 400) this.log.shift();
    // the new sentence's timing takes over only together with its audio: until then the clock (currentTime) is
    // still the previous clip's, and the word highlight would run ahead
    this.audio.src = url;
    if (state.cur?.item === item) {
      state.cur.timing = timing;
      state.cur.wi = 0;
      prepareTiming();
    }
    this.audio.playbackRate = 1; // speed is applied when the voice is synthesised (natural pitch)
    this.audio.play().catch(() => {});
    for (let k = 1; k <= 3; k++) if (this.queue[this.idx + k]) this.url(this.queue[this.idx + k].id).catch(() => {});
    this.topUp(g);
  },

  url(id) {
    // the sentence's audio and its word timing (for word-by-word highlighting), fetched together
    const key = `${id}@${state.speed}`;
    if (!this.urls.has(key)) {
      const q = `${encodeURIComponent(id)}?speed=${state.speed}`;
      this.urls.set(key, Promise.all([
        api(`/api/doc/${state.doc}/audio/${q}`).then((r) => r.blob()).then((b) => URL.createObjectURL(b)),
        api(`/api/doc/${state.doc}/timing/${q}`).then((r) => r.json()),
      ]).then(([url, timing]) => ({ url, timing })).catch((err) => {
        this.urls.delete(key); // do not keep a failure: the next attempt asks again
        throw err;
      }));
    }
    return this.urls.get(key);
  },

  async topUp(g) {
    // ask for the following sentences well before the queue runs out (they may cross into the next page)
    if (this.more || this.idx < this.queue.length - 12) return;
    const last = this.queue[this.queue.length - 1];
    this.more = fetchSequence(last.id).then((seq) => {
      if (g === this.gen) this.queue.push(...seq.slice(1));
    }).finally(() => { this.more = null; });
  },

  async next() {
    if (!this.playing) return;
    this.idx++;
    if (this.idx >= this.queue.length && this.more) await this.more;
    this.play(this.gen);
  },

  finish() {
    this.playing = false;
    setPlaying(false);
    status("Finished");
  },

  toggle() {
    if (!this.current) return startFromView();
    if (this.playing) {
      this.playing = false;
      this.audio.pause();
      setPlaying(false);
    } else {
      this.playing = true;
      setPlaying(true);
      if (this.audio.src && !this.audio.ended) this.audio.play();
      else this.play(++this.gen);
    }
  },

  step(delta) {
    if (!this.queue.length) return;
    this.idx = Math.max(0, Math.min(this.queue.length - 1, this.idx + delta));
    this.playing = true;
    this.audio.pause();
    this.play(++this.gen);
  },

  stop() {
    this.gen++;
    this.audio.pause();
    this.audio.removeAttribute("src");
    this.queue = [];
    this.current = null;
    this.playing = false;
    for (const u of this.urls.values()) u.then((x) => URL.revokeObjectURL(x.url)).catch(() => {});
    this.urls.clear();
    state.blockCursor = null;
    clearHighlight();
    setPlaying(false);
  },
};
player.audio.addEventListener("ended", () => { player.lastEnded = performance.now(); player.next(); });
player.audio.addEventListener("playing", () => {
  const last = player.log[player.log.length - 1];
  if (last && last.started == null) last.started = performance.now();
});
window.pdfReader = { player, state, highlight, render, norm, toPdf, clickContext, locateQuery }; // for debugging and automated tests (bench/)

function setPlaying(on) {
  $("play").textContent = on ? "⏸" : "▶";
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
}

function startFromView() {
  if (!state.pdf) return;
  const top = pagesEl.getBoundingClientRect().top;
  const p = state.pages.find((q) => q.div.getBoundingClientRect().bottom > top + 40) || state.pages[0];
  player.startAt({ page: p.n, x: 0, y: 0 });
}

// ------------------------------------------------------------------ input
function clickContext(p, target, clientX, clientY) {
  // what the server needs to find the clicked sentence (/locate): the point in PDF coordinates, the clicked line
  // of the text layer, the clicked character's offset in it (a line often ends one sentence and starts the next)
  const r = p.div.getBoundingClientRect();
  const { x, y } = toPdf(p, clientX - r.left, clientY - r.top);
  const span = target.matches(".textLayer span") ? target : null;
  const text = span ? span.textContent : "";
  let at = null, before = "", after = "";
  if (span) {
    const pos = document.caretPositionFromPoint?.(clientX, clientY);
    const rng = pos ? null : document.caretRangeFromPoint?.(clientX, clientY);
    const [node, off] = pos ? [pos.offsetNode, pos.offset] : rng ? [rng.startContainer, rng.startOffset] : [null, 0];
    if (node && span.contains(node)) {
      // a caret position is the gap between two characters: on the right half of a character it is the gap
      // after it, so the clicked character is the one before the gap when the point is inside that one's box
      at = off;
      if (off > 0 && node.nodeType === Node.TEXT_NODE) {
        const range = document.createRange();
        range.setStart(node, off - 1);
        range.setEnd(node, off);
        const b = range.getBoundingClientRect();
        if (clientX >= b.left && clientX <= b.right) at = off - 1;
      }
    }
    // the text around it: a short line ("M", a subscript) occurs many times in a paragraph, the stretch around
    // it once
    const k = p.textDivs.indexOf(span);
    for (let i = k - 1; i >= 0 && before.length < 40; i--) before = p.textDivs[i].textContent + " " + before;
    for (let i = k + 1; i < p.textDivs.length && after.length < 40; i++) after += " " + p.textDivs[i].textContent;
  }
  return { page: p.n, x, y, text, at, before, after };
}

pagesEl.addEventListener("click", (e) => {
  // Ctrl+click (Cmd+click on a Mac) reads from there; a plain click or drag selects and copies text as usual
  if (!(e.ctrlKey || e.metaKey)) return;
  const pageDiv = e.target.closest(".page");
  if (!pageDiv || !state.pdf) return;
  e.preventDefault();
  getSelection().removeAllRanges(); // Ctrl+click is not a selection
  const p = state.pages[+pageDiv.dataset.page - 1];
  if (!p.viewport) return;
  player.startAt(clickContext(p, e.target, e.clientX, e.clientY)).catch((err) => status(`Error: ${err.message}`));
});

// while Ctrl (Cmd) is held, the text shows a hand: a click will read from there
const ctrlCursor = (e) => document.body.classList.toggle("ctrl-read", e.ctrlKey || e.metaKey);
document.addEventListener("keydown", ctrlCursor);
document.addEventListener("keyup", ctrlCursor);
window.addEventListener("blur", () => document.body.classList.remove("ctrl-read"));

$("play").addEventListener("click", () => player.toggle());
$("prev").addEventListener("click", () => player.step(-1));
$("next").addEventListener("click", () => player.step(1));
$("follow").addEventListener("change", (e) => { state.follow = e.target.checked; });

// highlight: the sentence being read (default), the word being spoken, or nothing; kept for next time (this browser)
const HL_MODES = ["sentence", "word", "off"];
function setHighlight(mode) {
  if (!HL_MODES.includes(mode)) mode = "sentence";
  state.hlMode = mode;
  $("highlight").value = mode;
  pagesEl.classList.toggle("no-highlight", mode === "off");
  try { localStorage.setItem("papervoice.highlight", mode); } catch {}
  if (state.cur?.item) highlight(state.cur.item, false); // redraw the current sentence in the new mode
}
$("highlight").addEventListener("change", (e) => setHighlight(e.target.value));
let savedMode = "sentence";
try { savedMode = localStorage.getItem("papervoice.highlight") || "sentence"; } catch {}
setHighlight(savedMode);
$("speed").addEventListener("input", (e) => {
  state.speed = +(+e.target.value).toFixed(2);
  $("speedOut").textContent = `${state.speed.toFixed(2)}×`;
  // the current sentence finishes at the old speed; the next ones are fetched at the new one
  for (let k = 1; k <= 2; k++) if (player.queue[player.idx + k]) player.url(player.queue[player.idx + k].id).catch(() => {});
});
$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await openBytes(await f.arrayBuffer(), f.name).catch((err) => status(`Error: ${err.message}`));
});

// ------------------------------------------------------------------ zoom (redraws the pages at the new size)
const ZOOM_MIN = 0.4, ZOOM_MAX = 4;
let zoomTimer = null;

function setZoom(z) {
  state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, +z.toFixed(2)));
  $("zoomReset").textContent = `${Math.round(state.zoom * 100)}%`;
  clearTimeout(zoomTimer); // several steps in a row (wheel) are drawn once, when they stop
  zoomTimer = setTimeout(rebuildPages, 120);
}

async function rebuildPages() {
  if (!state.pdf) return;
  // keep the reading position: the page at the top of the view and how far down it is
  const top = pagesEl.getBoundingClientRect().top;
  const p = state.pages.find((q) => q.div.getBoundingClientRect().bottom > top) || state.pages[0];
  const frac = p ? (top - p.div.getBoundingClientRect().top) / p.div.offsetHeight : 0;
  for (const el of state.lit) el.remove();
  state.lit = [];
  state.mark = null;
  await buildPages();
  const q = state.pages[(p?.n || 1) - 1];
  if (q) pagesEl.scrollTop = q.div.offsetTop - pagesEl.offsetTop + frac * q.div.offsetHeight;
  if (player.current) {
    const cur = state.cur;
    if (cur) cur.p = state.pages[player.current.page - 1];
    highlight(player.current, false);
  }
}

$("zoomIn").addEventListener("click", () => setZoom(state.zoom * 1.1));
$("zoomOut").addEventListener("click", () => setZoom(state.zoom / 1.1));
$("zoomReset").addEventListener("click", () => setZoom(1));
pagesEl.addEventListener("wheel", (e) => { // Ctrl+wheel zooms the PDF, not the whole page
  if (!(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  setZoom(state.zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
}, { passive: false });

// a new window size, or a screen with another pixel density (browser zoom, second monitor): draw again
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(rebuildPages, 250); });
function watchPixelRatio() {
  matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener("change", () => {
    rebuildPages();
    watchPixelRatio();
  }, { once: true });
}
watchPixelRatio();

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea")) return;
  if ((e.ctrlKey || e.metaKey) && ["+", "=", "-", "0"].includes(e.key)) { // Ctrl +/-/0: zoom the PDF
    e.preventDefault();
    setZoom(e.key === "0" ? 1 : state.zoom * (e.key === "-" ? 1 / 1.1 : 1.1));
    return;
  }
  if (e.code === "Space") { e.preventDefault(); player.toggle(); }
  else if (e.key === "ArrowRight") player.step(1);
  else if (e.key === "ArrowLeft") player.step(-1);
  else if (e.key === "h" || e.key === "H") setHighlight(HL_MODES[(HL_MODES.indexOf(state.hlMode) + 1) % HL_MODES.length]);
  else if (e.key === "+" || e.key === "-") {
    const s = $("speed");
    s.value = (+s.value + (e.key === "+" ? 0.05 : -0.05)).toFixed(2);
    s.dispatchEvent(new Event("input"));
  }
});

pagesEl.addEventListener("dragover", (e) => { e.preventDefault(); pagesEl.classList.add("drop"); });
pagesEl.addEventListener("dragleave", () => pagesEl.classList.remove("drop"));
pagesEl.addEventListener("drop", async (e) => {
  e.preventDefault();
  pagesEl.classList.remove("drop");
  const f = [...e.dataTransfer.files].find((x) => x.type === "application/pdf" || x.name.endsWith(".pdf"));
  if (f) await openBytes(await f.arrayBuffer(), f.name).catch((err) => status(`Error: ${err.message}`));
});

// ------------------------------------------------------------------ prepare the page being looked at
// the engine prepares pages in the background from where the voice is; when the reader scrolls elsewhere and
// stops, the page in view goes to the front of the queue, so a Ctrl+click there finds it ready
let lookTimer = null, lookedAt = 0;
pagesEl.addEventListener("scroll", () => {
  clearTimeout(lookTimer);
  lookTimer = setTimeout(() => {
    if (!state.doc || !state.pages.length) return;
    const mid = pagesEl.getBoundingClientRect().top + pagesEl.clientHeight / 2;
    const p = state.pages.find((q) => q.div.getBoundingClientRect().bottom > mid);
    if (!p || p.n === lookedAt) return;
    lookedAt = p.n;
    fetch(`/api/doc/${state.doc}/page/${p.n}?wait=0`).catch(() => {});
  }, 600);
});

// ------------------------------------------------------------------ start
// while this page is open the reading engine stays running (it stops by itself after a while without any
// request, to free its memory; see papervoice-server --idle-exit)
setInterval(() => fetch("/api/health", { cache: "no-store" }).catch(() => {}), 60_000);

async function openDoc(doc, name) {
  // a document the browser extension has already sent to the server (?doc=<id>&name=<file>)
  status("Opening…");
  const buf = await (await api(`/api/doc/${encodeURIComponent(doc)}/pdf`)).arrayBuffer();
  await load({ doc, name: name || "document.pdf" }, buf);
}

const params = new URLSearchParams(location.search);
const path = params.get("path"); // ?path=C:/papers/x.pdf, handy for testing
if (params.get("doc")) {
  waitForServer().then(() => openDoc(params.get("doc"), params.get("name")))
    .catch((err) => status(`Error: ${err.message}`));
} else if (path) {
  waitForServer().then(() => openPath(path)).catch((err) => status(`Error: ${err.message}`));
} else {
  status("Open a PDF");
  // a file can be opened at once; this only reports when the reading engine is ready
  waitForServer().then(() => { if (!state.pdf) status("Ready · open a PDF"); });
}
