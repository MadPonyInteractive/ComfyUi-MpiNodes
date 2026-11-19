"""CPU check: MpiSaveVideo's optional alpha MASK writes a transparent VP9 .webm whose alpha IS the matte
(soft values kept), zeroes RGB only where alpha is 0, scales a smaller matte up, broadcasts one matte to
every frame, and leaves the plain .mp4 path alone when no alpha is connected (Cubric Studio MPI-1611,
the Video cut-out Flow).

    G:/ComfyUi/python_embeded/python.exe tests/test_save_video_alpha.py

Needs ComfyUI's folder_paths: set COMFYUI_DIR, default G:/ComfyUi/ComfyUI.
"""
import importlib
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "G:/ComfyUi/ComfyUI"))

pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
video = importlib.import_module("mpinodes.video")
FFMPEG = importlib.import_module("mpinodes.help_funcs").find_ffmpeg()

N, H, W = 9, 64, 96
work = tempfile.mkdtemp(prefix="mpi_alpha_")
video.folder_paths.get_output_directory = lambda: work

fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def decode_rgba(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-c:v", "libvpx-vp9", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgba", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 4)


# Flat mid-grey picture, so chroma subsampling does not blur RGB across the matte edge.
images = torch.full((N, H, W, 3), 0.6)
# Matte: left third 0, middle third a soft 0.5, right third 1.
matte = torch.zeros((N, H, W))
matte[:, :, W // 3: 2 * W // 3] = 0.5
matte[:, :, 2 * W // 3:] = 1.0

node = video.MpiSaveVideo()
out = node.doit(images, 24.0, "t_alpha", use_audio=False, alpha=matte)
v = out["ui"]["videos"][0]
path = out["result"][0]
check(path.endswith(".webm") and v["format"] == "video/webm", f"alpha -> .webm, format {v['format']}")
got = decode_rgba(path)
check(got.shape[0] == N, f"{got.shape[0]} frames back (want {N})")
a = got[..., 3].astype(int)
mid = slice(4, H - 4)
check(abs(a[:, mid, 4:W // 3 - 4].mean() - 0) < 3, f"alpha 0 region reads {a[:, mid, 4:W // 3 - 4].mean():.1f}")
check(abs(a[:, mid, W // 3 + 4: 2 * W // 3 - 4].mean() - 128) < 6, f"soft region keeps ~128: {a[:, mid, W // 3 + 4: 2 * W // 3 - 4].mean():.1f}")
check(abs(a[:, mid, 2 * W // 3 + 4: W - 4].mean() - 255) < 3, f"opaque region reads {a[:, mid, 2 * W // 3 + 4: W - 4].mean():.1f}")
rgb_cut = got[:, mid, 4:W // 3 - 4, :3].astype(int).mean()
rgb_soft = got[:, mid, W // 3 + 4: 2 * W // 3 - 4, :3].astype(int).mean()
check(rgb_cut < 10, f"RGB zeroed where alpha is 0: {rgb_cut:.1f}")
check(abs(rgb_soft - 153) < 10, f"RGB kept on the soft edge (~153): {rgb_soft:.1f}")

# One small matte for all frames: scaled up and broadcast.
small = torch.zeros((1, H // 2, W // 2))
small[:, :, W // 4:] = 1.0
out2 = node.doit(images, 24.0, "t_alpha_bcast", use_audio=False, alpha=small)
g2 = decode_rgba(out2["result"][0])
check(g2.shape[0] == N, f"broadcast matte -> {g2.shape[0]} frames")
check(g2[:, mid, 4:W // 2 - 6, 3].mean() < 3 and g2[:, mid, W // 2 + 6: W - 4, 3].mean() > 252,
      "small matte scaled to the frame size")

# Mismatched count raises.
try:
    node.doit(images, 24.0, "t_alpha_bad", use_audio=False, alpha=torch.ones((3, H, W)))
    check(False, "3 masks for 9 frames should raise")
except ValueError:
    check(True, "3 masks for 9 frames raises")

# No alpha: the old mp4 path, unchanged.
out3 = node.doit(images, 24.0, "t_plain", use_audio=False)
check(out3["result"][0].endswith(".mp4") and out3["ui"]["videos"][0]["format"] == "video/mp4", "no alpha -> .mp4 as before")
check(images.shape == (N, H, W, 3) and torch.equal(images, torch.full((N, H, W, 3), 0.6)), "input frames not mutated")

print("FAILED" if fails else "ALL PASS", len(fails))
sys.exit(1 if fails else 0)
