"""CPU checks for kj/ltx.py and kj/lora_extract.py (MPI-1341), tiny tensors, no weights. Where the
original comfyui-kjnodes pack is on disk each port is compared with the original on fixed inputs.
Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_kj_ltx.py

NOT covered (needs real LTX / model weights, check in a workflow run): MpiLTX2NAG.patch on a real
LTX 2 model (the caption projection / embeddings connector path), and MpiVAELoader loading a real
audio VAE. What IS covered: the NAG attention maths and the full cross-attention forward on a
tiny CrossAttention, the audio/video mask node, and LoRA extraction end to end on a fake patcher.
"""
import importlib
import importlib.util
import itertools
import os
import sys
import tempfile
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = COMFY + "/custom_nodes/comfyui-kjnodes"
sys.path.insert(0, COMFY)

import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_pkg("mpi_kj_port", os.path.join(ROOT, "kj"))
ltx = sys.modules["mpi_kj_port.ltx"]
lora = sys.modules["mpi_kj_port.lora_extract"]

orig_ltx = orig_lora = None
if os.path.isdir(ORIG):
    stub = types.ModuleType("kjorig")
    stub.__path__ = [ORIG]
    sys.modules["kjorig"] = stub
    import server  # type: ignore  # the original grabs PromptServer.instance at import; there is no server here
    server.PromptServer.instance = types.SimpleNamespace(send_sync=lambda *a, **k: None)
    def _load_orig(modname):
        spec = importlib.util.spec_from_file_location("kjorig.nodes." + modname, f"{ORIG}/nodes/{modname}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    orig_ltx = _load_orig("ltxv_nodes")
    orig_lora = _load_orig("lora_nodes")
    orig_nodes = _load_orig("nodes")


def rnd(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def same(a, b, what=""):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.equal(a, b), (what, float((a - b).abs().max()))


def close(a, b, what="", atol=1e-5):
    """Equal up to LAPACK rounding: full_matrices=False picks a different driver path, ~1e-7 apart."""
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.allclose(a.float(), b.float(), atol=atol), (what, float((a.float() - b.float()).abs().max()))


# ------------------------------------------------------------------ mask node

def _latents(with_masks, seed=1):
    video = {"samples": rnd(1, 4, 5, 3, 3, seed=seed)}      # B, C, T, H, W
    audio = {"samples": rnd(1, 2, 20, 4, seed=seed + 1)}    # B, C, T, F
    if with_masks:
        video["noise_mask"] = (rnd(1, 1, 5, 3, 3, seed=seed + 2) > 0.5).float()
        audio["noise_mask"] = (rnd(1, 2, 20, 4, seed=seed + 3) > 0.5).float()
    return video, audio


def _same_latent(a, b, what):
    if a is None or b is None:
        assert a is None and b is None, what
        return
    assert set(a) == set(b), what
    for k in a:
        same(a[k], b[k], f"{what} {k}")


def test_audio_video_mask():
    node = ltx.MpiLTXAudioVideoMask()
    # what flow_ltx_extend.json sets: fps 24, 0 -> 15 s video and audio, pad, add
    v, a = _latents(False)
    ov, oa = node.execute(24, 0, 15, 0, 15, "pad", "add", video_latent=v, audio_latent=a)
    assert ov["samples"].shape[2] == (15 * 24 - 1) // 8 + 1 and float(ov["noise_mask"].min()) == 1.0
    assert oa["samples"].shape[2] == 15 * 25 + 1 and float(oa["noise_mask"].min()) == 1.0
    assert v["samples"].shape[2] == 5  # inputs untouched
    # one side only
    assert node.execute(24, 0, 1, 0, 1, video_latent=v)[1] is None
    assert node.execute(24, 0, 1, 0, 1, audio_latent=a)[0] is None
    if orig_ltx:
        o = orig_ltx.LTXVAudioVideoMask
        n = 0
        for with_masks, mode, existing in itertools.product((False, True), ("truncate", "pad", "partial"), ("add", "subtract", "overwrite")):
            for (fps, vs, ve, as_, ae) in ((24, 0, 15, 0, 15), (24, 0.5, 1.2, 0.4, 0.6), (25, 0, 0.5, 0, 0.5), (12.5, 0.2, 3, 0.2, 3)):
                v, a = _latents(with_masks)
                mine = node.execute(fps, vs, ve, as_, ae, mode, existing, video_latent=v, audio_latent=a)
                v, a = _latents(with_masks)
                theirs = o.execute(fps, vs, ve, as_, ae, max_length=mode, existing_mask_mode=existing, video_latent=v, audio_latent=a).args
                _same_latent(mine[0], theirs[0], f"video {with_masks} {mode} {existing} {fps}")
                _same_latent(mine[1], theirs[1], f"audio {with_masks} {mode} {existing} {fps}")
                n += 1
        print(f"   audio/video mask: {n} parameter sets equal to the original")


# ------------------------------------------------------------------- NAG maths

def _ns(scale, alpha, tau, inplace):
    return types.SimpleNamespace(nag_scale=scale, nag_alpha=alpha, nag_tau=tau, inplace=inplace)


def test_nag_maths():
    pos, neg = rnd(2, 6, 16, seed=1), rnd(2, 6, 16, seed=2) * 3
    for inplace in (True, False):
        out = ltx.normalized_attention_guidance(_ns(11.0, 0.25, 2.5, inplace), pos.clone(), neg.clone())
        assert out.shape == pos.shape and torch.isfinite(out).all()
        # alpha 0 gives back the positive branch, whatever the scale
        zero = ltx.normalized_attention_guidance(_ns(11.0, 0.0, 2.5, inplace), pos.clone(), neg.clone())
        assert torch.allclose(zero, pos, atol=1e-6)
        # scale 1: guidance == positive, so no change at any alpha
        one = ltx.normalized_attention_guidance(_ns(1.0, 0.7, 2.5, inplace), pos.clone(), neg.clone())
        assert torch.allclose(one, pos, atol=1e-6)
        # tau clips the L1 norm ratio of the guidance to the positive branch
        big = ltx.normalized_attention_guidance(_ns(50.0, 1.0, 1.5, inplace), pos.clone(), neg.clone())
        ratio = big.abs().sum(-1) / pos.abs().sum(-1)
        assert float(ratio.max()) <= 1.5 + 1e-4
    if orig_ltx:
        for inplace, (s, al, t) in itertools.product((True, False), ((11.0, 0.25, 2.5), (3.0, 0.5, 10.0), (50.0, 1.0, 1.5))):
            a = ltx.normalized_attention_guidance(_ns(s, al, t, inplace), pos.clone(), neg.clone())
            b = orig_ltx.normalized_attention_guidance(_ns(s, al, t, inplace), pos.clone(), neg.clone())
            same(a, b, f"nag {inplace} {s} {al} {t}")


def _attn(gated, seed=0):
    import comfy.ops
    from comfy.ldm.lightricks.model import CrossAttention
    torch.manual_seed(seed)
    m = CrossAttention(query_dim=32, context_dim=32, heads=4, dim_head=8, apply_gated_attention=gated,
                       operations=comfy.ops.disable_weight_init).eval()
    for p in m.parameters():
        p.data.normal_(0, 0.5)
    return m


def test_nag_forward_matches_original():
    nag_ctx = rnd(1, 5, 32, seed=7) * 2
    for gated, batch in itertools.product((False, True), (1, 2)):
        m = _attn(gated)
        x, ctx = rnd(batch, 9, 32, seed=3), rnd(batch, 6, 32, seed=4)
        with torch.no_grad():
            mine = ltx.LTXVCrossAttentionPatch(nag_ctx, 11.0, 0.25, 2.5, inplace=True).__get__(m, type(m))(x, ctx)
            assert mine.shape == x.shape and torch.isfinite(mine).all()
            # it changes the plain attention (the positive half), and leaves the negative half plain
            plain = m(x, ctx)
            assert not torch.allclose(mine[:1], plain[:1])
            if batch == 2:
                assert torch.allclose(mine[1:], plain[1:], atol=1e-5)
            if orig_ltx:
                theirs = orig_ltx.LTXVCrossAttentionPatch(nag_ctx, 11.0, 0.25, 2.5, inplace=True).__get__(m, type(m))(x, ctx)
                same(mine, theirs, f"forward gated={gated} batch={batch}")
                mine2 = ltx.LTXVCrossAttentionPatch(nag_ctx, 5.0, 0.5, 3.0, inplace=False).__get__(m, type(m))(x, ctx)
                theirs2 = orig_ltx.LTXVCrossAttentionPatch(nag_ctx, 5.0, 0.5, 3.0, inplace=False).__get__(m, type(m))(x, ctx)
                same(mine2, theirs2, f"forward not inplace gated={gated} batch={batch}")


def test_nag_node_shape():
    cls = ltx.MpiLTX2NAG
    req, opt = cls.INPUT_TYPES()["required"], cls.INPUT_TYPES()["optional"]
    assert set(req) == {"model", "nag_scale", "nag_alpha", "nag_tau"} and set(opt) == {"nag_cond_video", "nag_cond_audio", "inplace"}
    assert req["nag_scale"][1]["default"] == 11.0 and req["nag_alpha"][1]["default"] == 0.25 and req["nag_tau"][1]["default"] == 2.5
    sentinel = object()
    assert cls().patch(sentinel, 0.0, 0.25, 2.5) == (sentinel,)  # scale 0 hands the model back


def test_vae_loader_shape():
    cls = ltx.MpiVAELoader
    req = cls.INPUT_TYPES()["required"]
    assert set(req) == {"vae_name", "device", "weight_dtype"}
    assert req["device"][0] == ["main_device", "cpu"] and req["weight_dtype"][0] == ["bf16", "fp16", "fp32"]
    assert cls.RETURN_TYPES == ("VAE",)
    if orig_ltx:
        o = orig_nodes.VAELoaderKJ.INPUT_TYPES()["required"]
        assert o["device"] == req["device"] and o["weight_dtype"] == req["weight_dtype"]
        # the original lists every vae file; ours lists exactly the vae folder
        assert set(req["vae_name"][0]) <= set(o["vae_name"][0])


# ----------------------------------------------------------------- LoRA extract

def test_extract_lora_matches_original():
    lin = torch.randn(24, 40, generator=torch.Generator().manual_seed(1))
    conv3 = torch.randn(12, 6, 3, 3, generator=torch.Generator().manual_seed(2))
    conv1 = torch.randn(12, 6, 1, 1, generator=torch.Generator().manual_seed(3))
    low_rank = torch.randn(24, 4, generator=torch.Generator().manual_seed(4)) @ torch.randn(4, 40, generator=torch.Generator().manual_seed(5))
    cases = []
    for diff in (lin, conv3, conv1, low_rank):
        for lora_type, algo in (("standard", "svd_linalg"), ("standard", "svd_lowrank"), ("adaptive_ratio", "svd_linalg"),
                                ("adaptive_energy", "svd_linalg"), ("adaptive_quantile", "svd_linalg"), ("adaptive_fro", "svd_linalg")):
            cases.append((diff, lora_type, algo))
    n = 0
    for diff, lora_type, algo in cases:
        if diff.ndim == 4 and algo == "svd_lowrank":
            continue  # the original raises NameError (lora_rank unset) for a conv weight here; the port fixes it
        for rank, clamp in ((8, False), (8, True), (64, False)):
            torch.manual_seed(0)
            mine = lora.extract_lora(diff, "k", rank, algo, lora_type, lowrank_iters=7, adaptive_param=0.4, clamp_quantile=clamp)
            if orig_lora:
                torch.manual_seed(0)
                theirs = orig_lora.extract_lora(diff, "k", rank, algo, lora_type, lowrank_iters=7, adaptive_param=0.4, clamp_quantile=clamp)
                for x, y in zip(mine, theirs):
                    close(x, y, f"{tuple(diff.shape)} {lora_type} {algo} r{rank} clamp{clamp}")
            n += 1
    # a low-rank diff is recovered at its rank
    u, vh = lora.extract_lora(low_rank, "k", 4, "svd_linalg", "standard", clamp_quantile=False)
    assert torch.allclose(u @ vh, low_rank, atol=1e-3)
    # the conv shapes the LoRA format wants
    u, vh = lora.extract_lora(conv3, "k", 4, "svd_linalg", "standard")
    assert u.shape == (12, 4, 1, 1) and vh.shape == (4, 6, 3, 3)
    # the original's svd_lowrank + conv NameError is fixed
    u, vh = lora.extract_lora(conv3, "k", 4, "svd_lowrank", "standard")
    assert u.shape == (12, 4, 1, 1)
    print(f"   extract_lora: {n} parameter sets equal to the original")


class _FakePatcher:
    """Stands in for a ModelPatcher whose weights were patched with (original - finetuned)."""

    def __init__(self, params, diffs):
        self.model = torch.nn.Module()
        for k in params:
            parent = self.model
            *path, leaf = k.split(".")
            for p in path:
                if not hasattr(parent, p):
                    parent.add_module(p, torch.nn.Module())
                parent = getattr(parent, p)
            parent.register_parameter(leaf, torch.nn.Parameter(params[k]))
        self.diffs = diffs

    def clone(self):
        return self

    def get_key_patches(self, prefix=""):
        return {}

    def add_patches(self, *a, **k):
        return []

    def patch_weight_to_device(self, key, return_weight=False):
        return self.diffs[key].clone()


def test_lora_extract_node():
    import folder_paths
    from safetensors.torch import load_file
    g = torch.Generator().manual_seed(11)
    names = ["diffusion_model.blocks.0.attn.weight", "diffusion_model.blocks.0.attn.bias", "diffusion_model.blocks.0.norm.weight",
             "diffusion_model.conv.weight"]
    shapes = [(16, 24), (16,), (16,), (8, 4, 3, 3)]
    params = {k: torch.zeros(s) for k, s in zip(names, shapes)}
    diffs = {k: torch.randn(*s, generator=g) for k, s in zip(names, shapes)}
    node = lora.MpiLoraExtract()
    assert node.OUTPUT_NODE and node.RETURN_TYPES == ()
    spec = node.INPUT_TYPES()["required"]
    assert spec["finetuned"][0] == "MODEL,CLIP" and spec["lora_type"][0][:2] == ["standard", "full"]
    # what Model Merger.json sets
    base = dict(filename_prefix="loras/kjtest", rank=4, lora_type="standard", algorithm="svd_linalg", lowrank_iters=7,
                output_dtype="fp16", bias_diff=True, adaptive_param=0.15, clamp_quantile=False)
    with tempfile.TemporaryDirectory() as d:
        orig_out = folder_paths.get_output_directory
        folder_paths.get_output_directory = lambda: d
        try:
            mine = _FakePatcher(params, diffs)
            node.extract(mine, mine, **base)
            files = sorted(os.listdir(os.path.join(d, "loras")))
            assert files == ["kjtest_rank_4_fp16_00001_.safetensors"], files
            got = load_file(os.path.join(d, "loras", files[0]))
            assert set(got) == {"diffusion_model.blocks.0.attn.lora_up.weight", "diffusion_model.blocks.0.attn.lora_down.weight",
                                "diffusion_model.blocks.0.attn.diff_b", "diffusion_model.blocks.0.norm.diff",
                                "diffusion_model.conv.lora_up.weight", "diffusion_model.conv.lora_down.weight"}, sorted(got)
            assert got["diffusion_model.blocks.0.attn.lora_up.weight"].dtype == torch.float16
            assert got["diffusion_model.blocks.0.attn.lora_up.weight"].shape == (16, 4)
            if orig_lora:
                mine = _FakePatcher(params, diffs)
                orig_lora.LoraExtractKJ.execute(mine, mine, **{**base, "filename_prefix": "loras/kjorig"})
                theirs = load_file(os.path.join(d, "loras", "kjorig_rank_4_fp16_00001_.safetensors"))
                assert set(theirs) == set(got)
                for k in got:
                    close(got[k], theirs[k], k, atol=2e-3)
            # an escape from the output folder is refused by folder_paths
            try:
                node.extract(mine, mine, **{**base, "filename_prefix": "../../escape"})
                raise AssertionError("expected the output-folder guard")
            except Exception as e:
                assert "outside" in str(e).lower(), e
            # svd_lowrank only exists for the standard type
            try:
                node.extract(mine, mine, **{**base, "algorithm": "svd_lowrank", "lora_type": "full"})
                raise AssertionError("expected ValueError")
            except ValueError:
                pass
        finally:
            folder_paths.get_output_directory = orig_out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all kj ltx/lora tests passed" + ("" if orig_ltx else " (original pack absent: equivalence skipped)"))
