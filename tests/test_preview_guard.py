"""preview_guard against the REAL ComfyUI latent_preview (Cubric Studio MPI-1467).

A file named taeh3* in vae_approx that is not madebyollin's taeh3 used to kill every MiniMax H3
sampler before its first step ('NoneType' object has no attribute 'show_progress_bar'). With the
guard it falls back to the colour preview; the real decoder still gives the TAEHV preview.

Needs a ComfyUI checkout (CPU is enough). Run with that ComfyUI's python:

    python tests/test_preview_guard.py <ComfyUI dir> [<real taeh3.safetensors>]

The real-decoder check is skipped when no taeh3 file is given.
"""
import importlib
import os
import shutil
import sys
import tempfile
import types

COMFY = os.path.abspath(sys.argv[1])
REAL_TAEH3 = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = [sys.argv[0], "--cpu"]  # ComfyUI parses argv on import
sys.path.insert(0, COMFY)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
import folder_paths  # noqa: E402
import comfy.latent_formats  # noqa: E402
from comfy.cli_args import args, LatentPreviewMethod  # noqa: E402
import latent_preview  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
guard = importlib.import_module("mpinodes.preview_guard")

args.preview_method = LatentPreviewMethod.TAESD
H3 = comfy.latent_formats.MiniMaxH3Video()
CPU = torch.device("cpu")


def use_vae_approx(folder):
    folder_paths.folder_names_and_paths["vae_approx"] = ([folder], folder_paths.supported_pt_extensions)
    folder_paths.filename_list_cache.clear()


def foreign_taeh3(folder):
    # Shaped like a 2D decoder-only TAE: no `decoder.22.bias`, so ComfyUI detects no VAE in it.
    save_file({"decoder.layers.1.weight": torch.zeros(64, 24, 3, 3)}, os.path.join(folder, "taeh3.safetensors"))


failures = 0


def check(name, ok, detail=""):
    global failures
    print(("PASS " if ok else "FAIL ") + name + (f" ({detail})" if detail else ""))
    failures += 0 if ok else 1


with tempfile.TemporaryDirectory() as d:
    foreign_taeh3(d)
    use_vae_approx(d)

    # The bug as reported, before the guard is installed.
    try:
        latent_preview.get_previewer(CPU, H3)
        check("unguarded core crashes on a foreign taeh3", False, "no error")
    except AttributeError as e:
        check("unguarded core crashes on a foreign taeh3", "show_progress_bar" in str(e), str(e))

    guard.install()
    guard.install()  # a second call must not wrap twice
    check("install() wraps once", not hasattr(latent_preview.get_previewer.__wrapped__, "__wrapped__"))

    p = latent_preview.get_previewer(CPU, H3)
    check("guarded: foreign taeh3 -> colour preview", isinstance(p, latent_preview.Latent2RGBPreviewer), type(p).__name__)

    # Other errors still raise: a file that is not a tensor file at all.
    with open(os.path.join(d, "taeh3.safetensors"), "wb") as f:
        f.write(b"not a safetensors file")
    folder_paths.filename_list_cache.clear()
    try:
        latent_preview.get_previewer(CPU, H3)
        check("guarded: unreadable file still raises", False, "no error")
    except guard.UnusableDecoder:
        check("guarded: unreadable file still raises", False, "swallowed as UnusableDecoder")
    except Exception as e:
        check("guarded: unreadable file still raises", True, type(e).__name__)

with tempfile.TemporaryDirectory() as d:
    use_vae_approx(d)
    p = latent_preview.get_previewer(CPU, H3)
    check("guarded: no taeh3 -> colour preview, as before", isinstance(p, latent_preview.Latent2RGBPreviewer), type(p).__name__)

if REAL_TAEH3:
    with tempfile.TemporaryDirectory() as d:
        shutil.copy(REAL_TAEH3, os.path.join(d, "taeh3.safetensors"))
        use_vae_approx(d)
        p = latent_preview.get_previewer(CPU, H3)
        check("guarded: real taeh3 -> TAEHV preview", isinstance(p, latent_preview.TAEHVPreviewerImpl), type(p).__name__)
else:
    print("SKIP real taeh3 (no file given)")

print("FAILED" if failures else "ALL PASS")
sys.exit(1 if failures else 0)
