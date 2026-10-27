// popup.js - send the PDF in the current tab to the local reading engine and open the read-along viewer.
// Permissions: activeTab (read the current tab's PDF, only when the icon is clicked) and the local server.
const SERVER = "http://127.0.0.1:8765";
const $ = (id) => document.getElementById(id);

function say(text) {
  $("message").textContent = text;
}

async function serverUp() {
  try {
    const r = await fetch(`${SERVER}/api/health`, { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

function fileName(url) {
  try {
    const u = new URL(url);
    const last = decodeURIComponent(u.pathname.split("/").filter(Boolean).pop() || "document");
    return last.toLowerCase().endsWith(".pdf") ? last : `${last}.pdf`;
  } catch {
    return "document.pdf";
  }
}

// the tab's own address; Chrome and Edge show PDFs at that address with their built-in viewer
function pdfUrl(tab) {
  const url = tab?.url || "";
  return /^(https?|file):/.test(url) ? url : null;
}

async function fetchPdf(url) {
  const r = await fetch(url, { credentials: "include" }); // signed-in sites (journals) need the cookies
  if (!r.ok) throw new Error(`the page answered ${r.status}`);
  const buf = await r.arrayBuffer();
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(5, buf.byteLength)));
  if (head !== "%PDF-") throw new Error("this tab is not showing a PDF");
  return buf;
}

async function read(tab) {
  const url = pdfUrl(tab);
  $("read").disabled = true;
  say("Sending the PDF to the reading engine…");
  try {
    const buf = await fetchPdf(url);
    const name = fileName(url);
    const r = await fetch(`${SERVER}/api/open?name=${encodeURIComponent(name)}`, {
      method: "POST", headers: { "Content-Type": "application/pdf" }, body: buf,
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `engine answered ${r.status}`);
    const info = await r.json();
    await chrome.tabs.create({ // the reader opens in a new tab next to the PDF
      url: `${SERVER}/?doc=${encodeURIComponent(info.doc)}&name=${encodeURIComponent(info.name || name)}`,
      index: tab.index + 1,
    });
    window.close();
  } catch (err) {
    $("read").disabled = false;
    if (url.startsWith("file:") && err instanceof TypeError) {
      $("file-hint").hidden = false;
      say("The extension may not read files from your disk yet.");
    } else {
      say(`Could not open it: ${err.message}.`);
    }
  }
}

// the reading engine is started by a small helper registered once with `uv run papervoice-setup`
// (native messaging: native_host.py); it stops by itself after a while without use
const HOST = "com.papervoice.reader";

async function startEngine() {
  // always asked, also when the engine answers: the helper replaces an engine still running older code
  let reply;
  try {
    reply = await chrome.runtime.sendNativeMessage(HOST, { cmd: "start" });
  } catch {
    return serverUp(); // helper not registered: an engine started by hand still works
  }
  if (reply?.running) return true;
  say(reply?.replaced ? "Updating the reading engine… (~15-30 s)" : "Starting the reading engine… (~15-30 s)");
  for (let i = 0; i < 120; i++) { // up to 2 minutes (the first start after installing downloads the models)
    if (await serverUp()) return true;
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

// Update notice. The only request this extension makes to the internet: version.json from the project's GitHub
// repository (a few bytes; GitHub sees the address like for any web page), at most every 12 hours, and it can be
// switched off below. Nothing about the user or their papers is sent. Updating itself is done by update.bat.
const VERSION_URL = "https://raw.githubusercontent.com/Gowravsharma/PaperVoice/main/version.json";
const CHECK_EVERY = 12 * 3600 * 1000;

function store(key, value) { // localStorage can be unavailable (private windows, blocked site data)
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { /* the check then simply runs each time */ }
  return null;
}

function newer(latest, installed) { // "1.10.0" > "1.9.2"
  const a = String(latest).split(".").map(Number), b = String(installed).split(".").map(Number);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const d = (a[i] || 0) - (b[i] || 0);
    if (d) return d > 0;
  }
  return false;
}

async function latestVersion() {
  const cached = JSON.parse(store("papervoice.latest") || "null");
  if (cached && Date.now() - cached.at < CHECK_EVERY) return cached.version;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 4000);
  try {
    const r = await fetch(VERSION_URL, { cache: "no-store", signal: ctl.signal });
    if (!r.ok) return cached?.version ?? null;
    const version = String((await r.json()).version || "");
    if (!/^\d+(\.\d+)*$/.test(version)) return null;
    store("papervoice.latest", JSON.stringify({ at: Date.now(), version }));
    return version;
  } catch {
    return cached?.version ?? null; // offline: no notice, nothing else changes
  } finally {
    clearTimeout(timer);
  }
}

async function showVersion() {
  const installed = chrome.runtime.getManifest().version;
  const on = store("papervoice.updatecheck") !== "off";
  $("version").textContent = `PaperVoice ${installed} · `;
  const toggle = document.createElement("a");
  toggle.textContent = on ? "update check: on" : "update check: off";
  toggle.title = "Looks for a new version on GitHub at most every 12 hours. Click to switch.";
  toggle.addEventListener("click", () => { store("papervoice.updatecheck", on ? "off" : "on"); location.reload(); });
  $("version").append(toggle);
  if (!on) return;
  const latest = await latestVersion();
  if (latest && newer(latest, installed)) {
    $("update").textContent = `Version ${latest} is available (you have ${installed}). To update: double-click ` +
      "update.bat in the PaperVoice folder, then reload the extension on the extensions page.";
    $("update").hidden = false;
  }
}

async function main() {
  showVersion(); // not awaited: it must never delay reading
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const up = await startEngine();
  $("server").className = `dot ${up ? "ok" : "bad"}`;
  $("server").title = up ? "Reading engine is running" : "Reading engine is not running";
  if (!up) {
    say("The reading engine could not be started.");
    $("help").hidden = false;
    $("help").open = true;
    return;
  }
  $("viewer").hidden = false;
  $("stop").hidden = false;
  $("stop").addEventListener("click", async () => {
    await fetch(`${SERVER}/api/shutdown`, { method: "POST" }).catch(() => {});
    say("Stopped. Its memory is free again; the next click starts it.");
    $("server").className = "dot bad";
    for (const id of ["read", "viewer", "stop"]) $(id).hidden = true;
  });
  $("viewer").addEventListener("click", () => { chrome.tabs.create({ url: `${SERVER}/` }); window.close(); });
  if (tab?.url?.startsWith(SERVER)) {
    say("This is the reader. Ctrl+click in the text to start reading from there.");
    return;
  }
  if (!pdfUrl(tab)) {
    say("Open a PDF in this tab, or choose a file in the reader.");
    return;
  }
  say(tab.url.toLowerCase().includes(".pdf") || tab.url.includes("/pdf/")
    ? "Ready to read this PDF." : "If this tab shows a PDF, it can be read aloud.");
  $("read").hidden = false;
  $("read").addEventListener("click", () => read(tab));
  $("read").focus();
}

main();
