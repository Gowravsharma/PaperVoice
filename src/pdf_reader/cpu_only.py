"""
cpu_only.py - answer Docling's "which device?" question without torch.

Docling's stages (layout engine, OCR, tables) call docling.utils.accelerator_utils.decide_device, which imports
torch to look for CUDA/MPS/XPU. This project installs no torch and runs on the CPU only, so the function is
replaced by one that says "cpu" - in accelerator_utils and in every Docling module that already imported it by
name. install() runs again before each converter is built, which also covers modules imported in between.
"""
import sys


def decide_device(accelerator_device, supported_devices=None):
    return "cpu"


def install():
    import docling.utils.accelerator_utils as au
    au.decide_device = decide_device
    for name, mod in list(sys.modules.items()):
        if name.startswith("docling") and getattr(mod, "decide_device", None) not in (None, decide_device):
            mod.decide_device = decide_device
