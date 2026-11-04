"""CPU checks for MpiUpscaleVideo (MPI-1418) and the shared VideoStream it and MpiRifeVideo use.

- the frames it encodes are the frames ImageUpscaleWithModel + ImageScaleBy make on the whole clip
  (the graph it replaces), model on and off;
- the mp4 has every frame, the source fps and the audio cut to the video;
- peak RAM does not grow with the clip length;
- MpiRifeVideo still makes (n-1)*m+1 frames through VideoStream.

Needs the bench ComfyUI and the 4x_NMKD-Siax_200k / rife47 weights. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_upscale_video.py
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types

import psutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
UPSCALE_MODEL = "M:/CubricModels/upscale_models/4x_NMKD-Siax_200k.pth"
sys.path.insert(0, COMFY)

import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")
import folder_paths  # noqa: E402

# The pack as a package without its __init__ (which imports every node).
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
uv = importlib.import_module("mpinodes.upscale_video")
vs_mod = importlib.import_module("mpinodes.video_stream")
video = importlib.import_module("mpinodes.video")
help_funcs = importlib.import_module("mpinodes.help_funcs")

WORK = tempfile.mkdtemp(prefix="mpi_upscale_video_")
folder_paths.set_input_directory(WORK)
folder_paths.set_output_directory(WORK)
FFMPEG = help_funcs.find_ffmpeg()


def make_clip(name, seconds, w, h, fps=12, audio=True):
    path = os.path.join(WORK, name)
    cmd = [FFMPEG, "-y", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds + 1}", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True, capture_output=True)
    return name


def load_upscale_model():
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    folder_paths.add_model_folder_path("upscale_models", os.path.dirname(UPSCALE_MODEL))
    out = UpscaleModelLoader.execute(os.path.basename(UPSCALE_MODEL))
    return out.result[0] if hasattr(out, "result") else out[0]


def reference(name, model, scale_by):
    """The old graph: MpiLoadVideoUpload -> ImageUpscaleWithModel -> ImageScaleBy(scale_by / model scale)."""
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    from nodes import ImageScaleBy
    path = os.path.join(WORK, name)
    _, w, h, _ = vs_mod.probe("test", path)
    frames, _ = video._decode_frames(FFMPEG, path, w, h)
    if model is not None:
        out = ImageUpscaleWithModel.execute(model, frames)
        frames = out.result[0] if hasattr(out, "result") else out[0]
        scale_by = scale_by / model.scale
    return ImageScaleBy().upscale(frames, "lanczos", scale_by)[0]


def run_captured(name, scale_by, model):
    """Run the node, keeping a uint8 copy of every frame it hands the encoder."""
    seen = []
    write = vs_mod.VideoStream.write

    def spy(self, frames):
        seen.append(frames.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).clone())
        return write(self, frames)

    vs_mod.VideoStream.write = spy
    try:
        res = uv.MpiUpscaleVideo().run("None", scale_by=scale_by, use_model=model is not None,
                                       string=name, upscale_model=model)
    finally:
        vs_mod.VideoStream.write = write
    return res, torch.cat(seen)


def mp4_facts(path):
    err = subprocess.run([FFMPEG, "-i", path], capture_output=True).stderr.decode("utf-8", "replace")
    _, w, h, fps = vs_mod.probe("test", path)
    frames, n = video._decode_frames(FFMPEG, path, w, h)
    a = subprocess.run([FFMPEG, "-i", path, "-map", "0:a:0", "-f", "null", "-"], capture_output=True).stderr.decode()
    import re
    t = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", a)
    audio_s = (int(t[-1][0]) * 3600 + int(t[-1][1]) * 60 + float(t[-1][2])) if t else None
    return w, h, fps, n, audio_s, "Audio:" in err


def to_uint8(images):
    return images.clamp(0, 1).mul(255).round().to(torch.uint8)


def test_lanczos_matches_graph():
    name = make_clip("a.mp4", 2, 64, 48)
    res, got = run_captured(name, 2.5, None)
    want = to_uint8(reference(name, None, 2.5))
    assert got.shape == want.shape == (24, 120, 160, 3), (got.shape, want.shape)
    assert torch.equal(got, want)
    w, h, fps, n, audio_s, has_audio = mp4_facts(res["result"][0])
    assert (w, h, n) == (160, 120, 24), (w, h, n)
    assert abs(fps - 12) < 0.01 and has_audio and abs(audio_s - 2.0) < 0.05, (fps, audio_s)
    assert res["ui"]["videos"][0]["filename"].endswith(".mp4")


def test_model_matches_graph():
    model = load_upscale_model()
    name = make_clip("b.mp4", 1, 48, 32, audio=False)
    res, got = run_captured(name, 2.0, model)
    want = to_uint8(reference(name, model, 2.0))
    assert got.shape == want.shape == (12, 64, 96, 3), (got.shape, want.shape)
    assert torch.equal(got, want), (got.int() - want.int()).abs().max()
    w, h, fps, n, _, has_audio = mp4_facts(res["result"][0])
    assert (w, h, n) == (96, 64, 12) and not has_audio, (w, h, n, has_audio)


def peak_rss(fn):
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    done = threading.Event()

    def watch():
        while not done.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            done.wait(0.01)

    t = threading.Thread(target=watch)
    t.start()
    try:
        fn()
    finally:
        done.set()
        t.join()
    return (peak[0] - base) / 2**20


def test_ram_flat():
    short = make_clip("short.mp4", 2, 320, 240, fps=24)
    long = make_clip("long.mp4", 20, 320, 240, fps=24)
    run = lambda n: uv.MpiUpscaleVideo().run("None", scale_by=4.0, string=n)
    run(short)  # warm-up: imports, first allocations
    s, l = peak_rss(lambda: run(short)), peak_rss(lambda: run(long))
    whole_clip_mb = 480 * 1280 * 960 * 3 * 4 / 2**20  # what the old graph held at the 4x output alone
    print(f"  peak RAM: 2 s clip +{s:.0f} MB, 20 s clip +{l:.0f} MB (old graph: >{whole_clip_mb:.0f} MB)")
    assert l - s < 100, (s, l)


def test_rife_through_videostream():
    stream = importlib.import_module("mpinodes.rife.stream")
    folder_paths.add_model_folder_path("rife", f"{COMFY}/models/rife")
    name = make_clip("c.mp4", 1, 64, 48)
    res = stream.MpiRifeVideo().run("rife47.pth", "None", multiplier=2, string=name)
    w, h, fps, n, audio_s, has_audio = mp4_facts(res["result"][0])
    assert (w, h, n) == (64, 48, 23) and abs(fps - 24) < 0.01, (w, h, n, fps)
    assert has_audio and abs(audio_s - 23 / 24) < 0.05, audio_s


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"ok {name}")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
