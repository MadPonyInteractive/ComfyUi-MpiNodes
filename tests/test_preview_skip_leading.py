"""MpiVideoSamplingPreview `skip_leading`: conditioning latents at the START are not previewed (Cubric Studio MPI-1601).

WanAnimate2ToVideo holds the reference picture as latent frame 0 (`trim_latent`). Previewed with the
clip it flashes the picture at each loop start and shifts the TAEHV frame grouping by one. With
skip_leading=1 the burst must be exactly what the clip alone decodes to; with 0 nothing changes.

Needs a ComfyUI checkout (CPU is enough) and a Wan 2.1 tiny decoder (taew2_1). Run with that ComfyUI's python:

    python tests/test_preview_skip_leading.py <ComfyUI dir> <taew2_1*.safetensors>
"""
import importlib
import os
import sys
import types

COMFY = os.path.abspath(sys.argv[1])
TAEW = sys.argv[2]
sys.argv = [sys.argv[0], "--cpu"]  # ComfyUI parses argv on import
sys.path.insert(0, COMFY)

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("mpinodes")
pkg.__path__ = [ROOT]
sys.modules["mpinodes"] = pkg
preview = importlib.import_module("mpinodes.preview")


class _Server:
    last_node_id = "41"
    client_id = None

    def __init__(self):
        self.sent = []

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data))


preview.mm.get_torch_device = lambda: torch.device("cpu")
vae = preview._load_tiny_vae(TAEW)
vae.first_stage_model.to("cpu")


def burst(x0, skip):
    srv = _Server()
    preview.server.PromptServer.instance = srv
    shown = []
    previewer = preview._TinyVaePreviewer(vae, 16, None, 0, skip)
    previewer._send = shown.append  # the decoded clip, before JPEG
    previewer.push(x0)
    return srv.sent[0][1]["length"], shown[0]


torch.manual_seed(0)
x0 = torch.randn(1, 16, 6, 8, 8)  # latent 0 = the reference, 1..5 = the clip

n_skip, f_skip = burst(x0, 1)
n_clip, f_clip = burst(x0[:, :, 1:], 0)
assert n_skip == n_clip == 17, (n_skip, n_clip)
diff = (f_skip - f_clip).abs().max().item()
assert diff < 1e-4, f"skip_leading=1 must burst the clip without its reference latent (max diff {diff})"
print("ok - skip_leading=1 bursts the clip alone:", n_skip, "frames")

keep = x0.clone()
n_all, f_all = burst(x0, 0)
assert torch.equal(keep, x0), "a preview must never write into the sampler's x0"
assert n_all == 21, n_all
print("ok - skip_leading=0 is unchanged: the whole latent,", n_all, "frames")

n_big, _ = burst(x0, 99)
assert n_big == 21, "a skip that would empty the clip is ignored, never a zero-frame preview"
print("ok - a skip >= the clip length is ignored")
