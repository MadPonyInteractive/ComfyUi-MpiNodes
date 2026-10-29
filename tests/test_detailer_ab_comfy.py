"""A/B of the port against the REAL Impact-Pack detailer inside the real ComfyUI code, on CPU with no weights.

A tiny random-weight UNet (comfy's own SD15 classes, 32 channels) plus a fake 8x VAE and a fake CLIP are put
through Impact's DetailerForEach.do_detail and through detailer.detail.do_detail with the same image, SEGS,
seed and settings; the output pictures must be bit-identical. That covers the sampler glue the other tests
cannot reach: noise, sigmas / denoise tail, schedulers, samplers (incl. an SDE one), cycles, differential
diffusion, the inpaint-model path, per-SEG wildcard prompts ([CONCAT][LAB], BREAK) and the paste.

Needs the G:/ComfyUi bench (comfy source + Impact pack + python). Run it on its own (it replaces `server`
and puts ComfyUI on sys.path):

    G:/ComfyUi/python_embeded/python.exe tests/test_detailer_ab_comfy.py
"""
import os
import sys
import types
from unittest import mock

COMFY = "G:/ComfyUi/ComfyUI"
IMPACT_DIR = f"{COMFY}/custom_nodes/comfyui-impact-pack/modules/impact"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if not os.path.isdir(IMPACT_DIR):
        return print("skip (no G:/ComfyUi bench)")

    sys.path.insert(0, COMFY)
    sys.path.append(REPO)  # appended: the repo's json.py must not shadow the stdlib
    sys.argv = ["comfy", "--cpu"]  # comfy.cli_args parses sys.argv

    import numpy as np
    import torch

    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models

    # the real Impact pack registers HTTP routes on a PromptServer instance at import: stub only `server`
    server = types.ModuleType("server")
    server.PromptServer = mock.MagicMock()
    sys.modules["server"] = server
    sys.path.insert(0, os.path.dirname(IMPACT_DIR))
    pkg = types.ModuleType("impact")
    pkg.__path__ = [IMPACT_DIR]
    sys.modules["impact"] = pkg
    sys.modules["impact.impact_server"] = mock.MagicMock()
    import impact.core as core
    import impact.impact_pack as pack

    from detailer import detail
    from detailer import segs as S

    torch.manual_seed(0)
    conf = comfy.supported_models.SD15({
        "context_dim": 32, "model_channels": 32, "use_linear_in_transformer": False, "adm_in_channels": None,
        "use_temporal_attention": False, "image_size": 32, "in_channels": 4, "out_channels": 4, "num_res_blocks": [1, 1],
        "channel_mult": (1, 2), "use_spatial_transformer": True, "legacy": False, "transformer_depth": [1, 1],
        "transformer_depth_output": [1, 1, 1, 1], "transformer_depth_middle": 1, "num_heads": 4, "num_head_channels": -1,
        "use_checkpoint": False, "dtype": torch.float32})
    conf.set_inference_dtype(torch.float32, None)
    base = comfy.model_base.BaseModel(conf, device=torch.device("cpu"))
    patcher = comfy.model_patcher.ModelPatcher(base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))

    class FakeVAE:
        """8x down / up like SD's VAE, deterministic."""

        def encode(self, pixels):
            x = torch.nn.functional.avg_pool2d(pixels[..., :3].permute(0, 3, 1, 2), 8)
            return torch.cat([x, x.mean(1, keepdim=True)], dim=1)

        def decode(self, latent):
            x = torch.nn.functional.interpolate(latent[:, :3], scale_factor=8, mode="nearest")
            return x.permute(0, 2, 3, 1).clamp(0, 1)

        def decode_tiled(self, latent, tile_x=64, tile_y=64):
            return self.decode(latent)

    class FakeCLIP:
        def tokenize(self, text):
            return {"t": text}

        def encode_from_tokens_scheduled(self, tokens, add_dict=None):
            g = torch.Generator().manual_seed(sum(map(ord, tokens["t"])) % 1000)
            return [[torch.randn(1, 4, 32, generator=g), {"pooled_output": torch.zeros(1, 32)}]]

    def cond(seed):
        g = torch.Generator().manual_seed(seed)
        return [[torch.randn(1, 4, 32, generator=g), {"pooled_output": torch.zeros(1, 32)}]]

    torch.manual_seed(1)
    image = torch.rand(1, 96, 128, 3)
    mask = np.zeros((96, 128), np.float32)
    mask[10:50, 12:60] = 1.0
    mask[40:90, 70:120] = 1.0
    segs = S.label_assign(S.mask_to_segs(mask, False, 1.5, False, 1, is_contour=False), ["t0", "t1"])
    rsegs = core.mask_to_segs(mask, False, 1.5, False, 1, is_contour=False)
    rsegs = (rsegs[0], [x._replace(label=label) for x, label in zip(rsegs[1], ["t0", "t1"])])

    cases = [
        dict(steps=3, denoise=0.5, sampler="euler", scheduler="simple", wildcard="", noise_mask=True, feather=5, nmf=0, inpaint=False, cycle=1, gs=64.0),
        dict(steps=4, denoise=0.4, sampler="heun", scheduler="beta", wildcard="[CONCAT][LAB][t0]a red door[t1]blue sky", noise_mask=True, feather=10, nmf=10, inpaint=False, cycle=1, gs=64.0),
        dict(steps=3, denoise=0.6, sampler="lcm", scheduler="normal", wildcard="plain words BREAK second chunk", noise_mask=False, feather=3, nmf=0, inpaint=False, cycle=2, gs=80.0),
        dict(steps=3, denoise=0.5, sampler="euler", scheduler="simple", wildcard="", noise_mask=True, feather=5, nmf=20, inpaint=True, cycle=1, gs=32.0),
        dict(steps=3, denoise=0.5, sampler="dpmpp_2m_sde", scheduler="karras", wildcard="", noise_mask=True, feather=5, nmf=0, inpaint=False, cycle=1, gs=64.0),
    ]
    for case in cases:
        rest = (patcher, FakeCLIP(), FakeVAE(), case["gs"], True, 96.0, 1234, case["steps"], 1.5, case["sampler"],
                case["scheduler"], cond(1), cond(2), case["denoise"], case["feather"], case["noise_mask"], True, case["wildcard"])
        kw = dict(cycle=case["cycle"], inpaint_model=case["inpaint"], noise_mask_feather=case["nmf"])
        mine = detail.do_detail(image, segs, *rest, **kw)
        theirs = pack.DetailerForEach.do_detail(image, rsegs, *rest, **kw)
        label = f"{case['sampler']}/{case['scheduler']} wildcard={case['wildcard'][:14]!r} inpaint={case['inpaint']} cycle={case['cycle']}"
        assert torch.equal(mine[0], theirs[0]), (label, (mine[0] - theirs[0]).abs().max().item())
        assert (mine[0] - image).abs().max() > 0.01, label  # the detailer really changed the picture
        assert len(mine[4][1]) == len(theirs[5][1]) == 2
        for a, b in zip(mine[3], theirs[3]):  # the enhanced crops with alpha
            assert torch.equal(a, b), label
        print("identical:", label, flush=True)
    print("AB OK")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # skip interpreter shutdown: ModelPatcher.__del__ complains about a torn-down comfy
