"""MpiVideoSamplingPreview on a STILL: a one-token H3 latent previews as ONE frame (Cubric Studio MPI-1455).

P-H3 Image samples a one-token latent. `_decode_clip` padded the decoded frames to a 20-frame chunk
and dropped a fixed 12-frame tail, which left the picture plus 4 frames of its OWN zero pad; the app
looped them as picture, black, black, black, black. A clip keeps its old frame count.

Needs a ComfyUI checkout (CPU is enough) and madebyollin's taeh3. Run with that ComfyUI's python:

    python tests/test_preview_h3_still.py <ComfyUI dir> <taeh3.safetensors>
"""
import importlib
import os
import sys
import types

COMFY = os.path.abspath(sys.argv[1])
TAEH3 = sys.argv[2]
sys.argv = [sys.argv[0], "--cpu"]  # ComfyUI parses argv on import
sys.path.insert(0, COMFY)

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
preview = importlib.import_module("mpinodes.preview")

vae = preview._load_tiny_vae(TAEH3)
model = vae.first_stage_model.to("cpu")
dtype = model.decoder[1].weight.dtype


def frames(tokens):
    torch.manual_seed(0)
    latent = torch.randn(1, 24, tokens, 8, 8, dtype=dtype)
    return preview._decode_clip(model, latent)[0]  # [3,T,H,W]


def old_count(tokens):
    """The frame count the code had before this fix."""
    decoded = tokens * model.t_upscale
    chunk = 5 * model.t_upscale
    padded = decoded + (-decoded % chunk)
    return padded // chunk * (chunk - model.frames_to_trim) - 3 * model.t_upscale


still = frames(1)
assert still.shape[1] == 1, f"a still must preview as 1 frame, got {still.shape[1]}"
assert still.abs().amax() > 0, "the one frame is the picture, not pad"
print("ok - one token -> 1 frame (was", old_count(1), "with", old_count(1) - 1, "black pad frames)")

for tokens in (17, 25):
    clip = frames(tokens)
    assert clip.shape[1] == old_count(tokens), f"{tokens} tokens: {clip.shape[1]} != {old_count(tokens)}"
    blank = [i for i in range(clip.shape[1]) if clip[:, i].abs().amax() == 0]
    assert not blank, f"{tokens} tokens: all-zero frames at {blank}"
    print(f"ok - {tokens} tokens -> {clip.shape[1]} frames, unchanged, none blank")
