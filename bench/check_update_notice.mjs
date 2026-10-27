// check_update_notice.mjs - the extension popup's "update available" notice, in headless Edge with the real
// extension loaded. GitHub's answer is replaced through the DevTools protocol (Fetch domain), and starting the
// reading engine (native messaging) is stubbed, so nothing here touches the real engine or the network.
//
// Usage: node bench/check_update_notice.mjs <edge.exe> <fresh profile dir> <extension dir>
import { spawn } from "node:child_process";

const [edge, profile, extDir] = process.argv.slice(2);
const ID = "fininfbbcceiipeicgnhagiefcldlcom"; // fixed by the key in manifest.json
const port = 9344;
const proc = spawn(edge, ["--headless=new", "--disable-gpu", `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`, `--load-extension=${extDir}`, `--disable-extensions-except=${extDir}`,
  "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 40; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(250); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map(), errors = [];
let mode = "newer", asked = 0;
ws.addEventListener("message", async (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  if (msg.method === "Runtime.exceptionThrown") errors.push(msg.params.exceptionDetails.exception?.description);
  if (msg.method === "Fetch.requestPaused") {
    const rid = msg.params.requestId;
    asked++;
    if (mode === "offline") send("Fetch.failRequest", { requestId: rid, errorReason: "InternetDisconnected" });
    else {
      const version = mode === "newer" ? "1.0.1" : mode === "same" ? "1.0.0" : mode === "bigger" ? "1.10.0" : "0.9.9";
      send("Fetch.fulfillRequest", { requestId: rid, responseCode: 200,
        responseHeaders: [{ name: "Content-Type", value: "application/json" }, { name: "Access-Control-Allow-Origin", value: "*" }],
        body: Buffer.from(JSON.stringify({ version })).toString("base64") });
    }
  }
});
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result.result.value;

await send("Runtime.enable");
await send("Page.enable");
await send("Fetch.enable", { patterns: [{ urlPattern: "https://raw.githubusercontent.com/*" }] });
// the popup asks the helper to start the engine: answer "already running" instead
await send("Page.addScriptToEvaluateOnNewDocument", { source:
  "if (window.chrome && chrome.runtime) chrome.runtime.sendNativeMessage = async () => ({ ok: true, running: true });" });

const url = `chrome-extension://${ID}/popup.html`;
async function open(label) {
  await send("Page.navigate", { url });
  await sleep(1800);
  return evaluate(`({ hidden: document.getElementById("update").hidden, text: document.getElementById("update").textContent,
    footer: document.getElementById("version").textContent })`);
}
let pass = 0, fail = 0;
const check = (name, ok, detail = "") => { ok ? pass++ : fail++; console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  (" + detail + ")" : ""}`); };

await send("Page.navigate", { url }); await sleep(1500);
await evaluate("localStorage.clear()");

mode = "newer"; asked = 0;
let r = await open();
check("popup shows its installed version", /PaperVoice 1\.0\.0/.test(r.footer), r.footer);
check("a newer version on GitHub (1.0.1) shows the notice", !r.hidden && /1\.0\.1/.test(r.text) && /update\.bat/.test(r.text), r.text.slice(0, 90));
check("one request was made", asked === 1, `${asked}`);

asked = 0; mode = "same";
r = await open();
check("opened again within 12 h: no new request (cached)", asked === 0, `${asked} requests`);
check("  ...and the notice still shows from the cache", !r.hidden);

await evaluate("localStorage.removeItem('papervoice.latest')");
asked = 0; mode = "same";
r = await open();
check("same version (1.0.0): no notice", r.hidden && asked === 1);

await evaluate("localStorage.removeItem('papervoice.latest')");
mode = "bigger"; r = await open();
check("1.10.0 counts as newer than 1.0.0 (numbers, not text)", !r.hidden && /1\.10\.0/.test(r.text), r.text.slice(0, 40));

await evaluate("localStorage.removeItem('papervoice.latest')");
mode = "older"; r = await open();
check("an older version on GitHub: no notice", r.hidden);

await evaluate("localStorage.clear()");
mode = "offline"; asked = 0; r = await open();
check("offline: no notice and the popup still works", r.hidden && /PaperVoice 1\.0\.0/.test(r.footer));

await evaluate("localStorage.clear(); localStorage.setItem('papervoice.updatecheck', 'off')");
mode = "newer"; asked = 0; r = await open();
check("update check switched off: no request, no notice", asked === 0 && r.hidden, `${asked} requests`);
check("  ...and the footer says it is off", /off/.test(r.footer), r.footer);

// the switch itself: a click flips it
await evaluate("document.querySelector('#version a').click()");
await sleep(1800);
const stored = await evaluate("localStorage.getItem('papervoice.updatecheck')");
check("clicking the switch turns the check back on", stored === "on", `stored: ${stored}`);

console.log(`${pass} passed, ${fail} failed; page errors: ${errors.length ? errors : "none"}`);
ws.close();
proc.kill();
process.exit(fail ? 1 : 0);
