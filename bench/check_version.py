"""
check_version.py - the places that name the version must agree before a release:
  pyproject.toml            the Python package
  extension/manifest.json   what the popup shows as installed
  version.json              what the popup compares against (the copy on GitHub's main branch is "the latest")
  uv.lock                   records the package's version too: `uv sync --locked` (the installer) refuses a lock
                            that differs, so after changing pyproject.toml run `uv lock`

Usage: uv run python bench/check_version.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
found = {
    "pyproject.toml": re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1),
    "extension/manifest.json": json.loads((ROOT / "extension" / "manifest.json").read_text(encoding="utf-8"))["version"],
    "uv.lock": re.search(r'name = "papervoice"\s+version = "([^"]+)"', (ROOT / "uv.lock").read_text(encoding="utf-8")).group(1),
    "version.json": json.loads((ROOT / "version.json").read_text(encoding="utf-8"))["version"],
}
for name, v in found.items():
    print(f"  {v:10} {name}")
if len(set(found.values())) != 1:
    sys.exit("the versions differ: set all of them to the version being released")
print("all agree")
