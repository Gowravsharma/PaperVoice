"""
doctor.py - can this computer run PaperVoice's native libraries? (`papervoice-doctor`)

Usage:
    uv run papervoice-doctor        (the installer runs it right after installing the Python parts)

Each library with native code (numpy, onnxruntime, opencv, ...) is imported in a process of its own, so one that
crashes the process outright (Windows "access violation", exit code -1073741819 / 0xC0000005, which Python cannot
catch) is reported instead of ending the check. It also prints what is needed to understand a failure: the processor
and its AVX instructions, and the Microsoft C++ runtime files these libraries are built against.

If onnxruntime crashes because the C++ runtime in System32 is too old (found on a Windows 10 computer with 14.26), it
puts a newer copy into the PaperVoice folder instead of asking the user to change the computer (vc_runtime.py), checks
again, and checks all libraries together in the order the engine imports them.
Exit code 0: everything loads. 1: something does not (the report says what, and what to try).
"""
import ctypes
import glob
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from pdf_reader import vc_runtime

# the libraries with native code that PaperVoice imports (onnxruntime runs every model, including the voice)
LIBRARIES = ["numpy", "onnxruntime", "cv2", "PIL", "pypdfium2", "docling_parse", "tokenizers", "kaldifst",
             "scipy", "pandas", "shapely", "lxml", "piper"]
# the order the engine imports them in (a library that loads a runtime file first decides which copy all get)
ENGINE_ORDER = ["pypdfium2", "docling_parse", "cv2", "shapely", "scipy", "pandas", "kaldifst", "wetext", "tokenizers",
                "PIL", "onnxruntime", "piper"]
RUNTIME_FILES = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140_1.dll"]
VC_REDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
NEEDED = (14, 40)  # the runtime the current libraries are built against (Visual Studio 17.10 and later)
ACCESS_VIOLATION = 0xC0000005


def _powershell(command):
    run = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60)
    return run.stdout.strip()


def _file_version(path):
    return _powershell(f"(Get-Item '{path}').VersionInfo.ProductVersion")


def _version_tuple(text):
    try:
        return tuple(int(x) for x in text.strip().split(".")[:2])
    except ValueError:
        return (0, 0)


def cpu_report():
    lines = [f"processor : {_powershell('(Get-CimInstance Win32_Processor).Name') or platform.processor()}"]
    present = ctypes.windll.kernel32.IsProcessorFeaturePresent
    # PF_AVX_INSTRUCTIONS_AVAILABLE = 39, PF_AVX2_INSTRUCTIONS_AVAILABLE = 40, PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
    feats = {name: bool(present(n)) for name, n in (("AVX", 39), ("AVX2", 40), ("AVX-512F", 41))}
    lines.append("instructions: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in feats.items()))
    lines.append(f"Windows   : {platform.platform()}")
    lines.append(f"Python    : {sys.version.split()[0]} ({sys.executable})")
    return lines, feats


def system32():
    return os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32")


def runtime_report():
    lines = []
    for name in RUNTIME_FILES:
        path = os.path.join(system32(), name)
        if os.path.exists(path):
            lines.append(f"{name:20} {_file_version(path)}")
        else:
            lines.append(f"{name:20} MISSING from System32")
    if vc_runtime.RUNTIME_FILE.exists():
        lines.append(f"{'msvcp140.dll':20} {_file_version(vc_runtime.RUNTIME_FILE)}  (PaperVoice's own copy, used first)")
    return lines


def system_runtime_version():
    path = os.path.join(system32(), "msvcp140.dll")
    return _version_tuple(_file_version(path)) if os.path.exists(path) else (0, 0)


def newer_runtime_in_numpy():
    """numpy ships a recent msvcp140.dll (under a hashed name) inside its own package: its path, if new enough."""
    spec = importlib.util.find_spec("numpy")
    if not spec or not spec.origin:
        return None
    libs = os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "numpy.libs", "msvcp140*.dll")
    for path in glob.glob(libs):
        if _version_tuple(_file_version(path)) >= NEEDED:
            return path
    return None


def try_import(names):
    """'ok', or a description of how importing failed, in a process of its own. `import pdf_reader` comes first, so
    the private C++ runtime (if there is one) is loaded before the libraries, exactly as in every PaperVoice process."""
    run = subprocess.run([sys.executable, "-c", f"import pdf_reader, {names}"], capture_output=True, text=True,
                         timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if run.returncode == 0:
        return "ok"
    code = run.returncode & 0xFFFFFFFF
    if code == ACCESS_VIOLATION:
        return "CRASHED (access violation, 0xC0000005)"
    if run.returncode < 0 or code > 0x80000000:
        return f"CRASHED (exit code 0x{code:08X})"
    last = (run.stderr.strip().splitlines() or ["no message"])[-1]
    return f"FAILED to import: {last}"


def check_libraries():
    bad = {}
    for name in LIBRARIES:
        result = try_import(name)
        print(f"  {name:14} {result}")
        if result != "ok":
            bad[name] = result
    return bad


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
    bad = check_libraries()

    fixed = False
    if any("0xC0000005" in r for r in bad.values()) and not vc_runtime.RUNTIME_FILE.exists():
        have = system_runtime_version()
        if have < NEEDED:
            print(f"\nThe C++ runtime in System32 is {have[0]}.{have[1]}; the libraries need {NEEDED[0]}.{NEEDED[1]} or newer.")
            print("Trying a private copy of a newer one inside the PaperVoice folder (nothing system-wide changes)...")
            source = newer_runtime_in_numpy()
            if source:
                vc_runtime.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, vc_runtime.RUNTIME_FILE)
                print("\nLibraries with native code, again:")
                bad = check_libraries()
                fixed = not bad
                if bad:  # it did not help: leave the folder as it was
                    vc_runtime.RUNTIME_FILE.unlink(missing_ok=True)
            else:
                print("(no newer copy was found inside the installed packages)")

    if not bad:
        together = try_import(", ".join(ENGINE_ORDER))
        print(f"  {'all together':14} {together}  (in the order the engine imports them)")
        if together != "ok":
            bad["all together"] = together

    print()
    if not bad:
        if fixed:
            print("Fixed: the C++ runtime of this computer is too old, so PaperVoice uses its own newer copy")
            print(f"({vc_runtime.RUNTIME_FILE}). Nothing outside the PaperVoice folder was changed.\n")
        print("Everything loads. PaperVoice can run on this computer.")
        return 0
    print("PROBLEM: " + ", ".join(bad) + (" cannot be loaded." if len(bad) > 1 else " cannot be loaded."))
    if any("0xC0000005" in r for r in bad.values()):
        print("A native library crashed while loading. A known cause of this is an out-of-date or conflicting")
        print("Microsoft Visual C++ runtime, so the next thing to try is the current one (Microsoft's own")
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
