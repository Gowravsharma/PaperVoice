"""
doctor.py - can this computer run PaperVoice's native libraries? (`papervoice-doctor`)

Usage:
    uv run papervoice-doctor        (the installer runs it right after installing the Python parts)

Each library with native code (numpy, onnxruntime, opencv, ...) is imported in a process of its own, so one that
crashes the process outright (Windows "access violation", exit code -1073741819 / 0xC0000005, which Python cannot
catch) is reported instead of ending the check. It also prints what is needed to understand a failure: the processor
and its AVX instructions, and the Microsoft C++ runtime files these libraries are built against.
Exit code 0: everything loads. 1: something does not (the report says what, and what to try).
"""
import ctypes
import os
import platform
import subprocess
import sys

# the libraries with native code that PaperVoice imports (onnxruntime runs every model, including the voice)
LIBRARIES = ["numpy", "onnxruntime", "cv2", "PIL", "pypdfium2", "docling_parse", "tokenizers", "kaldifst",
             "scipy", "pandas", "shapely", "lxml", "piper"]
RUNTIME_FILES = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140_1.dll"]
VC_REDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
ACCESS_VIOLATION = 0xC0000005


def _powershell(command):
    run = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60)
    return run.stdout.strip()


def cpu_report():
    lines = [f"processor : {_powershell('(Get-CimInstance Win32_Processor).Name') or platform.processor()}"]
    present = ctypes.windll.kernel32.IsProcessorFeaturePresent
    # PF_AVX_INSTRUCTIONS_AVAILABLE = 39, PF_AVX2_INSTRUCTIONS_AVAILABLE = 40, PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
    feats = {name: bool(present(n)) for name, n in (("AVX", 39), ("AVX2", 40), ("AVX-512F", 41))}
    lines.append("instructions: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in feats.items()))
    lines.append(f"Windows   : {platform.platform()}")
    lines.append(f"Python    : {sys.version.split()[0]} ({sys.executable})")
    return lines, feats


def runtime_report():
    lines = []
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in RUNTIME_FILES:
        path = os.path.join(system32, name)
        if os.path.exists(path):
            version = _powershell(f"(Get-Item '{path}').VersionInfo.ProductVersion")
            lines.append(f"{name:20} {version}")
        else:
            lines.append(f"{name:20} MISSING from System32")
    return lines


def try_import(name):
    """'ok', or a description of how the import failed, from a process of its own."""
    run = subprocess.run([sys.executable, "-c", f"import {name}"], capture_output=True, text=True, timeout=180,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if run.returncode == 0:
        return "ok"
    code = run.returncode & 0xFFFFFFFF
    if code == ACCESS_VIOLATION:
        return "CRASHED (access violation, 0xC0000005)"
    if run.returncode < 0 or code > 0x80000000:
        return f"CRASHED (exit code 0x{code:08X})"
    last = (run.stderr.strip().splitlines() or ["no message"])[-1]
    return f"FAILED to import: {last}"


def main():
    if os.name != "nt":
        print("papervoice-doctor checks the Windows setup; nothing to check here.")
        return 0
    print("Checking this computer for PaperVoice...\n")
    cpu, feats = cpu_report()
    print("\n".join(cpu))
    print("\nMicrosoft C++ runtime (System32):")
    print("\n".join("  " + line for line in runtime_report()))
    print("\nLibraries with native code:")
    bad = {}
    for name in LIBRARIES:
        result = try_import(name)
        print(f"  {name:14} {result}")
        if result != "ok":
            bad[name] = result
    print()
    if not bad:
        print("Everything loads. PaperVoice can run on this computer.")
        return 0
    print("PROBLEM: " + ", ".join(bad) + (" cannot be loaded." if len(bad) > 1 else " cannot be loaded."))
    if any("0xC0000005" in r for r in bad.values()):
        print("A native library crashed while loading. A known cause of this is an out-of-date or conflicting")
        print("Microsoft Visual C++ runtime, so the first thing to try is the current one (Microsoft's own")
        print("installer, an update of a shared runtime; it asks for administrator permission):")
        print(f"    {VC_REDIST}")
        print("Then restart Windows and run this check again (papervoice-doctor). If it still fails, send")
        print("everything printed above - it shows exactly which library and which runtime files are involved.")
    elif not feats.get("AVX"):
        print("This processor has no AVX instructions, which the machine-learning library needs.")
    else:
        print("Send everything printed above; it shows which library fails and why.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
