"""CPU check: the shared video decode (_decode_cmd, used by MpiLoadVideo, MpiLoadVideoUpload and VideoStream)
reads a clip at its true colour, so a graph that loads a video and saves it again does not darken it.

ffmpeg's default swscale read yuv420p ~1.1-1.6 % dark per channel and an extend chain compounded it every pass
(Cubric Studio MPI-1437). Four decode -> MpiSaveVideo-style encode round trips must stay within 0.6 % of the source.

    G:/ComfyUi/python_embeded/python.exe tests/test_video_decode.py

Needs ComfyUI's folder_paths: set COMFYUI_DIR, default G:/ComfyUi/ComfyUI.
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "G:/ComfyUi/ComfyUI"))

# The pack as a package without its __init__ (which imports every node).
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
video = importlib.import_module("mpinodes.video")
FFMPEG = importlib.import_module("mpinodes.help_funcs").find_ffmpeg()

W, H = 384, 672
ACCURATE = ["-sws_flags", "accurate_rnd+full_chroma_int"]


def frames(cmd):
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def mean_rgb(path):
    """The reference reading: an accurate decode, independent of _decode_cmd."""
    cmd = [FFMPEG, "-v", "error", "-i", path] + ACCURATE + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return frames(cmd).mean(axis=(0, 1, 2), dtype=np.float64)


def encode(rgb, path):
    """MpiSaveVideo's encode settings."""
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24",
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", path],
                   input=rgb.tobytes(), check=True)


def drift_after(passes, decode_cmd, work):
    """% change per channel after `passes` decode -> encode round trips of a noisy grey clip."""
    src = os.path.join(work, "src.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=0x5a5a50:s={W}x{H}:d=1:r=24,noise=alls=12:allf=t",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", src], check=True)
    base, cur = mean_rgb(src), src
    for i in range(passes):
        nxt = os.path.join(work, f"pass{i + 1}.mp4")
        encode(frames(decode_cmd(cur)), nxt)
        cur = nxt
    return 100 * (mean_rgb(cur) / base - 1)


def test_round_trips_keep_the_colour():
    work = tempfile.mkdtemp(prefix="mpi_video_decode_")
    try:
        drift = drift_after(4, lambda p: video._decode_cmd(FFMPEG, p), work)
        assert np.all(np.abs(drift) < 0.6), f"_decode_cmd drifts {np.round(drift, 2)} % over 4 round trips"
        # The test can see the fault: the default swscale decode darkens the same clip.
        old = drift_after(4, lambda p: [FFMPEG, "-i", p, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], work)
        assert np.all(old < -2.0), f"default decode should darken, got {np.round(old, 2)} %"
        print(f"  _decode_cmd 4 passes {np.round(drift, 2)} %, default decode {np.round(old, 2)} %")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_force_rate_keeps_the_flags():
    cmd = video._decode_cmd(FFMPEG, "x.mp4", force_rate=24)
    assert cmd[cmd.index("-sws_flags") + 1] == "accurate_rnd+full_chroma_int"
    assert cmd[cmd.index("-vf") + 1] == "fps=24"
    assert cmd[-5:] == ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
