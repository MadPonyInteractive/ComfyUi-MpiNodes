"""CPU check: a video whose AAC sound stops short of its last frame loads with the sound padded to the picture,
so the H3 extend audio splice lands (Cubric Studio MPI-1660).

A user's 174-frame clip at 24 fps carried 319488 samples (312 AAC blocks), 237 short of 7.25 s. The extend graph
cuts the original's sound to its last frame, appends the new window's sound and splices the re-take back at
frame_count - 24; with the short track every run raised "runs 237 samples past the end".

    G:/ComfyUi/python_embeded/python.exe tests/test_video_audio_length.py

Needs ComfyUI's folder_paths: set COMFYUI_DIR, default G:/ComfyUi/ComfyUI.
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("COMFYUI_DIR", "G:/ComfyUi/ComfyUI"))

pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
video = importlib.import_module("mpinodes.video")
FFMPEG = importlib.import_module("mpinodes.help_funcs").find_ffmpeg()

FRAMES, FPS, RATE = 174, 24.0, 44100


def short_sound_clip(work):
    """174 frames at 24 fps with an AAC track cut 5 ms before the picture ends."""
    src = os.path.join(work, "short_sound.mp4")
    subprocess.run([FFMPEG, "-v", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=s=64x64:r=24:d={FRAMES / FPS}",
                    "-f", "lavfi", "-i", f"sine=f=440:r={RATE}:d=7.2445",
                    "-frames:v", str(FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-ac", "2", src], check=True)
    return src


def extend_splice(original):
    """The flow_h3_extend.json audio chain, with a stand-in H3 window decode of 141 frames."""
    import torch  # type: ignore

    rng, splice = video.MpiAudioRange(), video.MpiAudioSplice()
    channels = original["waveform"].shape[1]
    h3 = {"waveform": torch.rand((1, channels, round(141 * RATE / FPS))) * 0.1, "sample_rate": RATE}
    head = rng.doit(original, FPS, 0, FRAMES - 1)[0]          # node 950
    tail = rng.doit(h3, FPS, 39, -1)[0]                       # node 942, context_frames = 39
    track = {"waveform": torch.cat([head["waveform"], tail["waveform"]], -1), "sample_rate": RATE}  # AudioConcat
    patch = rng.doit(h3, FPS, 39 - 24, -1)[0]                 # node 951
    return splice.doit(track, patch, FPS, FRAMES - 24, 800)[0]  # node 952, start = frame_count - 24


def test_short_sound_is_padded_and_the_splice_lands():
    import folder_paths  # type: ignore

    work = tempfile.mkdtemp(prefix="mpi_audio_len_", dir=folder_paths.get_temp_directory())
    try:
        src = short_sound_clip(work)
        raw = video._load_audio(FFMPEG, src)
        want = round(FRAMES * RATE / FPS)
        short = want - raw["waveform"].shape[-1]
        assert short > RATE // 1000, f"fixture should be short by more than the splice's 1 ms, is {short}"

        # The fault is visible: the decoded track as-is fails the splice the way the report did.
        try:
            extend_splice(raw)
            raise AssertionError("the unpadded track should have raised")
        except ValueError as e:
            assert "samples past the end" in str(e), e

        out = video.MpiLoadVideo().load(src, block_if_empty=False)
        audio, n, has_audio = out[1], out[3], out[7]
        assert n == FRAMES and has_audio
        assert audio["waveform"].shape[-1] == want, f"{audio['waveform'].shape[-1]} vs {want}"
        assert audio["waveform"][..., -short:].abs().max() == 0, "the padding must be silence"
        assert (audio["waveform"][..., :raw["waveform"].shape[-1]] == raw["waveform"]).all(), "sound unchanged"
        extend_splice(audio)
        print(f"  decoded {raw['waveform'].shape[-1]} samples, padded {short} to {want}; splice lands")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_long_and_missing_sound_are_left_alone():
    import torch  # type: ignore

    longer = {"waveform": torch.ones((1, 2, 400000)), "sample_rate": RATE}
    assert video._pad_to_picture(longer, FRAMES, FPS) is longer
    assert video._pad_to_picture(None, FRAMES, FPS) is None
    assert video._pad_to_picture(longer, FRAMES, 0.0) is longer


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
