"""Exercise MpiAudioRange / MpiAudioSplice index resolution with no ComfyUI and no GPU.

Run it with the ComfyUI portable interpreter, which has torch:

    <ComfyUI>/python_embeded/python.exe check_audio_range.py

The case that matters is the one that broke a real run: a negative start used to be
resolved against a track length ROUNDED to whole frames, so "from the end" landed up
to half a frame past the end. At 44100 Hz / 24 fps a frame is 1837.5 samples, so a
real soundtrack is almost never a whole number of frames.
"""
import os, sys, types, tempfile

PACK = os.path.dirname(os.path.abspath(__file__))

# Same synthetic-package import as check_splat.py: the pack uses relative imports,
# and putting PACK on sys.path would let the pack's json.py shadow the stdlib.
_pkg = types.ModuleType("mpinodes")
_pkg.__path__ = [PACK]
sys.modules["mpinodes"] = _pkg

fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: tempfile.gettempdir()
fp.get_input_directory = fp.get_temp_directory = fp.get_output_directory
sys.modules["folder_paths"] = fp
sys.modules["comfy"] = types.ModuleType("comfy")

import importlib
import torch

video = importlib.import_module("mpinodes.video")
R, S = video.MpiAudioRange(), video.MpiAudioSplice()
RATE, FPS = 44100, 24.0


def track(n, ch=2):
    return {"waveform": torch.randn(1, ch, n), "sample_rate": RATE}


# --- THE FAILING CASE, exact numbers from the first end-to-end Extend Video run:
#     a 229688-sample patch at start -125 on a 366625-sample track (199.52 frames).
full = track(366625)
patch = track(229688)
out = S.doit(full, patch, FPS, -125, 0)[0]["waveform"]
assert out.shape[-1] == 366625
assert torch.equal(out[..., 136937:], patch["waveform"][:1]), "patch must end exactly at the track end"
assert torch.equal(out[..., :136937], full["waveform"][:1, :, :136937]), "nothing before the patch is touched"
print("ok  -125 on a 199.52-frame track lands at sample 136937 and ends on the last sample")

# --- The rewired re-take start (context 39 - 24 = 15 frames) is 27562.5 samples, so
#     the patch cut at +15 and its -(L-15) placement round 1 sample apart. That
#     sample is dropped, not raised on.
gen = track(259088)  # 141 frames of generated audio
cut = R.doit(gen, FPS, 15, -1)[0]
assert cut["waveform"].shape[-1] == 259088 - 27562
joined = track(366625)
out = S.doit(joined, cut, FPS, 15 - 141, 0)[0]["waveform"]
assert out.shape[-1] == 366625
print("ok  a 1-sample rounding overhang is dropped, not raised")

# --- A genuinely different window (a whole frame out) still raises.
try:
    S.doit(joined, track(229688 + 1838), FPS, -125, 0)
    raise AssertionError("a patch a frame too long must raise")
except ValueError as e:
    assert "past the end" in str(e)
print("ok  a patch a whole frame too long still raises")

# --- Cut then write back with the same negative start: the same samples, bit-exact.
for n in (366625, 367500, 366626):  # fractional, whole, and just-over frame counts
    t = track(n)
    for start in (-125, -1, -24):
        w = R.doit(t, FPS, start, -1)[0]
        i0 = n - round(-start * RATE / FPS)
        assert torch.equal(w["waveform"], t["waveform"][..., i0:]), (n, start)
        back = S.doit(t, w, FPS, start, 0)[0]["waveform"]
        assert torch.equal(back, t["waveform"][:1]), (n, start)
print("ok  a negative cut spliced straight back is a bit-exact no-op")

# --- -1 reaches the LAST SAMPLE, not the end of the last whole frame (199.4 frames
#     used to round DOWN and drop the tail).
n = round(199.4 * RATE / FPS)
assert R.doit(track(n), FPS, 0, -1)[0]["waveform"].shape[-1] == n
# A positive end still cuts at a frame boundary: this is how a padded track is trimmed.
assert R.doit(track(n), FPS, 0, 97)[0]["waveform"].shape[-1] == round(98 * RATE / FPS)
# Positive starts are unchanged: frame 16 is sample 29400.
assert R.doit(track(259088), FPS, 16, -1)[0]["waveform"].shape[-1] == 259088 - 29400
# An empty or inverted range is empty, never an error.
assert R.doit(track(n), FPS, 50, 10)[0]["waveform"].shape[-1] == 0
assert R.doit(track(n), FPS, 5000, -1)[0]["waveform"].shape[-1] == 0
print("ok  -1 is the last sample; positive frames and empty ranges behave")

print("ALL OK")
