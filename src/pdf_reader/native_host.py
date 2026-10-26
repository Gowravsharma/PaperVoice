"""
native_host.py - lets the browser extension start the reading engine, so nobody has to run the server by hand.

    uv run papervoice-setup            # once: register this helper with Chrome and Edge (current user, no admin)
    uv run papervoice-setup --remove   # undo

Browsers allow an extension to start a program only through "native messaging": a registered helper that the
browser launches and talks to over stdin/stdout (4-byte length + JSON). This helper understands one message,
{"cmd": "start"}: if the reading engine is not answering on 127.0.0.1:8765 it starts papervoice-server in the
background (no window, detached from the browser so it keeps running), and replies. The server stops by itself
after 20 minutes without use (server.py --idle-exit), freeing its memory; the next click starts it again.

Registration writes, for the current user only: the helper's launcher and manifest in %LOCALAPPDATA%\\PaperVoice,
and one registry key per browser (HKCU\\Software\\<browser>\\NativeMessagingHosts\\com.papervoice.reader) pointing
to the manifest. Only the extension with ID EXTENSION_ID (fixed by the "key" in extension/manifest.json) may use it.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOST_NAME = "com.papervoice.reader"
EXTENSION_ID = "fininfbbcceiipeicgnhagiefcldlcom"
SERVER_URL = "http://127.0.0.1:8765"
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "out" / "server.log"
HOME = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PaperVoice"
OLD_HOME = HOME.with_name("pdf-reader")  # where versions before the rename to PaperVoice put the same two files
REGISTRY = (r"Software\Google\Chrome\NativeMessagingHosts", r"Software\Microsoft\Edge\NativeMessagingHosts",
            r"Software\BraveSoftware\Brave-Browser\NativeMessagingHosts")


def code_version():
    """Identifies the installed code: the size and time of every Python file of the package. A server started
    before an update reports an older value, and is replaced (serve: {"cmd": "start"})."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).resolve().parent.rglob("*.py")):
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def server_health():
    try:
        with urllib.request.urlopen(SERVER_URL + "/api/health", timeout=2) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def server_running():
    return server_health() is not None


def stop_server():
    req = urllib.request.Request(SERVER_URL + "/api/shutdown", data=b"", method="POST")
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except OSError:
        pass
    for _ in range(50):  # until the port is free
        if not server_running():
            return
        time.sleep(0.2)


def start_server():
    """papervoice-server in the background: no console window, its own process group, outside the browser's
    job object (where allowed), output to out/server.log."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    cmd = [str(pythonw if pythonw.exists() else exe), "-m", "pdf_reader.server"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LOG, "ab")
    flags = 0
    if os.name == "nt":
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000))
    kwargs = dict(cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
    try:
        subprocess.Popen(cmd, creationflags=flags, **kwargs)
    except OSError:  # the browser's job object may not allow breaking away
        subprocess.Popen(cmd, creationflags=flags & ~0x01000000, **kwargs)


def read_message():
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        return None
    return json.loads(sys.stdin.buffer.read(struct.unpack("<I", raw)[0]))


def send_message(obj):
    data = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(data)) + data)
    sys.stdout.buffer.flush()


def serve():
    """The helper as the browser runs it (one message per call from the extension)."""
    while (msg := read_message()) is not None:
        if msg.get("cmd") == "start":
            health = server_health()
            if health is not None and health.get("code") == code_version():
                send_message({"ok": True, "running": True})
            else:
                if health is not None:  # running code from before an update: replace it
                    stop_server()
                start_server()
                send_message({"ok": True, "started": True, "replaced": health is not None})
        else:
            send_message({"ok": False, "error": f"unknown command {msg.get('cmd')!r}"})


def remove_old_home():
    """The launcher and manifest an earlier version wrote under the old name (only those two files)."""
    for f in (OLD_HOME / f"{HOST_NAME}.json", OLD_HOME / "native-host.bat"):
        f.unlink(missing_ok=True)
    try:
        OLD_HOME.rmdir()  # only if nothing else is in it
    except OSError:
        pass


def install():
    remove_old_home()
    HOME.mkdir(parents=True, exist_ok=True)
    launcher = HOME / "native-host.bat"
    # the browser runs this file; it must print nothing but the helper's messages
    launcher.write_text(f'@echo off\r\n"{sys.executable}" -m pdf_reader.native_host %*\r\n', encoding="utf-8")
    manifest = HOME / f"{HOST_NAME}.json"
    manifest.write_text(json.dumps({
        "name": HOST_NAME, "description": "Starts the PaperVoice reading engine (papervoice-server)",
        "path": str(launcher), "type": "stdio", "allowed_origins": [f"chrome-extension://{EXTENSION_ID}/"],
    }, indent=2), encoding="utf-8")
    import winreg
    for base in REGISTRY:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, f"{base}\\{HOST_NAME}") as k:
            winreg.SetValueEx(k, "", 0, winreg.REG_SZ, str(manifest))
    print(f"registered {HOST_NAME} for Chrome, Edge and Brave (current user): {manifest}")
    print("The extension now starts the reading engine by itself. Reload the extension if it is already loaded.")


def remove():
    import winreg
    for base in REGISTRY:
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, f"{base}\\{HOST_NAME}")
        except FileNotFoundError:
            pass
    for f in (HOME / f"{HOST_NAME}.json", HOME / "native-host.bat"):
        f.unlink(missing_ok=True)
    try:
        HOME.rmdir()
    except OSError:
        pass
    remove_old_home()
    print(f"removed {HOST_NAME}")


def setup():
    ap = argparse.ArgumentParser(description="Let the browser extension start the reading engine (once).")
    ap.add_argument("--remove", action="store_true", help="undo the registration")
    a = ap.parse_args()
    if os.name != "nt":
        sys.exit("papervoice-setup registers the helper on Windows; on macOS/Linux start papervoice-server yourself "
                 "for now (reports welcome)")
    remove() if a.remove else install()


if __name__ == "__main__":
    serve()
