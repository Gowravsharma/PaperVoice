"""
vc_runtime.py - PaperVoice's own copy of Microsoft's C++ runtime, for computers where the one in System32 is too old.

Some Windows 10 computers have a Visual C++ runtime from 2019-2020 in System32 (msvcp140.dll 14.26). onnxruntime,
which runs every model, is built against a newer one and crashes while loading there (an "access violation", exit
code -1073741819). Installing Microsoft's current runtime fixes it but changes the whole computer and needs
administrator rights; this does not: when `papervoice-doctor` sees that crash, it puts a newer msvcp140.dll (the copy
numpy ships inside its own package) into tools/runtime/ in the PaperVoice folder, and `preload()` loads it first in
every PaperVoice process. Windows then hands the same module to every library that asks for "msvcp140.dll", whatever
the order they are imported in - which matters: kaldifst loads System32's before onnxruntime does, and a copy placed
only next to onnxruntime would lose. If tools/runtime/msvcp140.dll does not exist (every healthy computer) this does
nothing at all.
"""
import ctypes
import os
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[2] / "tools" / "runtime"
RUNTIME_FILE = RUNTIME_DIR / "msvcp140.dll"


def preload():
    """Load PaperVoice's private msvcp140.dll, if there is one. Never raises."""
    if os.name != "nt" or not RUNTIME_FILE.exists():
        return
    try:
        ctypes.WinDLL(str(RUNTIME_FILE))
    except OSError:  # unreadable or wrong architecture: the System32 one is used, as without this module
        pass
