"""The sound of a clip must survive a chain of video ops (MPI-1603).

A user's extend chain -> interpolate -> upscale came out with audibly degraded sound: every step
decoded the AAC track and encoded it again at ffmpeg's default ~128k, so the losses stacked.

- the streaming nodes (MpiUpscaleVideo, MpiRifeVideo through VideoStream) COPY the source's audio
  packets: the output's packets are the source's, only cut to the new length;
- MpiSaveVideo, which must encode a waveform, keeps the sound close to the source after several
  saves in a row (printed beside what the old ~128k default gave).

Needs the bench ComfyUI (shares tests/test_upscale_video.py's setup). Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_audio_generations.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_upscale_video as t  # noqa: E402  (the bench setup: folder_paths, the pack as a package)

FFMPEG, WORK, video, uv = t.FFMPEG, t.WORK, t.video, t.uv
GENERATIONS = 6


def packet_hashes(path):
    """Per-packet md5s of the first audio stream, copied out, never decoded."""
    out = subprocess.run([FFMPEG, "-i", path, "-map", "0:a:0", "-c", "copy", "-f", "framemd5", "-"],
                         capture_output=True, check=True).stdout.decode()
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def audio_codec(path):
    err = subprocess.run([FFMPEG, "-i", path], capture_output=True).stderr.decode("utf-8", "replace")
    return next((l.split("Audio:")[1].split()[0].strip(",") for l in err.splitlines() if "Audio:" in l), None)


def test_stream_copies_aac():
    name = t.make_clip("copy.mp4", 2, 64, 48)
    res = uv.MpiUpscaleVideo().run("None", scale_by=2.0, string=name)
    src, out = packet_hashes(os.path.join(WORK, name)), packet_hashes(res["result"][0])
    assert len(out) >= len(src) - 2 and out == src[: len(out)], (len(src), len(out))
    _, _, _, n, audio_s, _ = t.mp4_facts(res["result"][0])
    assert n == 24 and abs(audio_s - 2.0) < 0.05, (n, audio_s)


def test_stream_falls_back_when_mp4_cannot_hold_the_codec():
    """A .mkv with Vorbis audio: mp4 cannot take the packets, so the track is encoded once, at the high rate."""
    name = "vorbis.mkv"
    subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libvorbis", os.path.join(WORK, name)],
                   check=True, capture_output=True)
    res = uv.MpiUpscaleVideo().run("None", scale_by=2.0, string=name)
    assert audio_codec(res["result"][0]) == "aac", audio_codec(res["result"][0])
    _, _, _, n, audio_s, _ = t.mp4_facts(res["result"][0])
    assert n == 24 and abs(audio_s - 2.0) < 0.05, (n, audio_s)


def music(seconds=4, rate=44100):
    """Stereo tones, a sweep and a little noise: enough high content for AAC loss to show."""
    x = np.arange(seconds * rate) / rate
    rng = np.random.default_rng(0)
    left = 0.25 * np.sin(2 * np.pi * 220 * x) + 0.15 * np.sin(2 * np.pi * (500 + 1500 * x) * x) + 0.03 * rng.standard_normal(x.size)
    right = 0.25 * np.sin(2 * np.pi * 330 * x) + 0.1 * np.sin(2 * np.pi * 6000 * x) + 0.03 * rng.standard_normal(x.size)
    return {"waveform": torch.from_numpy(np.stack([left, right])).float()[None], "sample_rate": rate}


def snr_db(ref, got):
    """SNR of `got` against `ref`, after lining them up (AAC may shift by its priming)."""
    a, b = ref[0].numpy(), got[0].numpy()
    probe = 4096
    lags = range(-2048, 2049)
    best = max(lags, key=lambda k: float(np.dot(a[0, probe:probe * 3], b[0, probe + k:probe * 3 + k])))
    n = min(a.shape[1], b.shape[1] - best) - probe
    a, b = a[:, probe:probe + n], b[:, probe + best:probe + best + n]
    return 10 * np.log10(np.sum(a ** 2) / np.sum((a - b) ** 2))


def generations(bitrate):
    video.AUDIO_BITRATE = bitrate
    clip = torch.zeros((96, 32, 32, 3))
    audio = src = music()
    for _ in range(GENERATIONS):
        path = video.MpiSaveVideo().doit(clip, 24.0, "gen", audio=audio)["result"][0]
        audio = video._load_audio(FFMPEG, path)
    return snr_db(src["waveform"], audio["waveform"])


def test_save_keeps_sound_over_generations():
    keep = video.AUDIO_BITRATE
    try:
        old, new = generations("128k"), generations(keep)
    finally:
        video.AUDIO_BITRATE = keep
    print(f"  SNR after {GENERATIONS} saves: {old:.1f} dB at 128k (old default), {new:.1f} dB at {keep}")
    assert new > old + 3, (old, new)


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"ok {name}")
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
