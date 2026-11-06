"""CPU checks for MpiLoraExtractFiles (kj/lora_extract.py, Cubric Studio MPI-1452): tiny fake
safetensors files, no real weights. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_lora_extract_files.py

Covered: a diffusion-only base against an all-in-one checkpoint fine-tune (prefix stripped, VAE /
text-encoder keys ignored), scaled-fp8 dequant (scalar and per-row scale), the exact low-rank
reconstruction, the output keys and dtype, the refusals (int8, wrong architecture, not a fine-tune
of this base, not .safetensors). NOT covered: a real model's keys mapping onto ComfyUI's loaded
model (that is the Wan 5B run in Cubric Studio MPI-1452 validation).
"""
import importlib.util
import os
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
sys.path.insert(0, COMFY)

import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")
import folder_paths  # noqa: E402


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load_pkg("mpi_kj_port", os.path.join(ROOT, "kj"))
lora = sys.modules["mpi_kj_port.lora_extract"]

G = torch.Generator().manual_seed(7)
SHAPES = {"blocks.0.attn.weight": (16, 24), "blocks.0.attn.bias": (16,), "blocks.0.norm.weight": (16,),
          "conv.weight": (8, 4, 3, 3), "patch.weight": (8, 4, 1, 2, 2)}
RANK = 3


def _base():
    return {k: torch.randn(*s, generator=G) for k, s in SHAPES.items()}


def _low_rank_delta(shape):
    """A delta that rank RANK reproduces exactly (conv: on its flattened 2D view)."""
    out = shape[0]
    rest = int(torch.tensor(shape[1:]).prod()) if len(shape) > 1 else 1
    if len(shape) < 2:
        return torch.randn(*shape, generator=G) * 0.1
    return ((torch.randn(out, RANK, generator=G) @ torch.randn(RANK, rest, generator=G)) * 0.1).reshape(shape)


class _Env:
    """Fake diffusion_models / checkpoints / output folders in a temp dir."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.dirs = {f: os.path.join(d, f) for f in ("diffusion_models", "checkpoints", "output")}
        for p in self.dirs.values():
            os.makedirs(p)
        self._saved = (folder_paths.get_filename_list, folder_paths.get_full_path, folder_paths.get_output_directory)
        folder_paths.get_filename_list = lambda f: sorted(os.listdir(self.dirs[f])) if f in self.dirs else []
        folder_paths.get_full_path = lambda f, n: (os.path.join(self.dirs[f], n)
                                                   if f in self.dirs and os.path.isfile(os.path.join(self.dirs[f], n)) else None)
        folder_paths.get_output_directory = lambda: self.dirs["output"]
        return self

    def put(self, folder, name, sd):
        save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(self.dirs[folder], name))

    def run(self, ft, base, **kw):
        args = dict(filename_prefix="loras/t", rank=RANK, algorithm="svd_linalg", lowrank_iters=7,
                    output_dtype="fp32", bias_diff=True, clamp_quantile=False)
        args.update(kw)
        res = lora.MpiLoraExtractFiles().extract(ft, base, **args)
        item = res["ui"]["lora"][0]
        return load_file(os.path.join(self.dirs["output"], item["subfolder"], item["filename"])), item

    def __exit__(self, *a):
        folder_paths.get_filename_list, folder_paths.get_full_path, folder_paths.get_output_directory = self._saved
        self.tmp.cleanup()


def _check_reconstruction(got, base, ft_dense):
    for k in SHAPES:
        delta = ft_dense[k] - base[k]
        stem = "diffusion_model." + k.rsplit(".", 1)[0]
        if k.endswith(".bias"):
            assert torch.allclose(got[stem + ".diff_b"], delta, atol=1e-5), k
        elif len(SHAPES[k]) == 5:
            assert not any(x.startswith(stem + ".") for x in got), f"5D {k} should be skipped"
        elif len(SHAPES[k]) < 2:
            assert torch.allclose(got[stem + ".diff"], delta, atol=1e-5), k
        else:
            up, down = got[stem + ".lora_up.weight"], got[stem + ".lora_down.weight"]
            rec = (up.flatten(1) @ down.flatten(1)).reshape(delta.shape)
            assert torch.allclose(rec, delta, atol=1e-4), (k, (rec - delta).abs().max())


def test_checkpoint_finetune_vs_bare_base():
    base = _base()
    ft = {k: v + _low_rank_delta(v.shape) for k, v in base.items()}
    ckpt = {"model.diffusion_model." + k: v for k, v in ft.items()}
    ckpt["first_stage_model.decoder.weight"] = torch.randn(4, 4, generator=G)  # VAE: ignored
    ckpt["cond_stage_model.x.weight"] = torch.randn(4, 4, generator=G)  # text encoder: ignored
    with _Env() as env:
        env.put("diffusion_models", "base.safetensors", base)
        env.put("checkpoints", "ft.safetensors", ckpt)
        assert lora.MpiLoraExtractFiles.INPUT_TYPES()["required"]["finetuned_name"][0] == ["base.safetensors", "ft.safetensors"]
        got, item = env.run("ft.safetensors", "base.safetensors")
        assert item["subfolder"] == "loras" and item["filename"] == "t_rank_3_fp32_00001_.safetensors", item
        assert not any("first_stage" in k or "cond_stage" in k for k in got)
        _check_reconstruction(got, base, ft)
        got16, _ = env.run("ft.safetensors", "base.safetensors", output_dtype="fp16")
        assert all(v.dtype == torch.float16 for v in got16.values())


def test_scaled_fp8_dequant():
    f8 = torch.float8_e4m3fn
    base = _base()
    ft_dense, ft_file = {}, {}
    for i, (k, v) in enumerate(base.items()):
        target = v + _low_rank_delta(v.shape)
        if k.endswith(".weight") and len(v.shape) >= 2:
            if i % 2:  # scalar scale (ComfyUI scaled_fp8)
                scale = target.abs().max() / 448.0
                q = (target / scale).to(f8)
                ft_file[k[:-6] + "scale_weight"] = scale.reshape(1)
                ft_dense[k] = q.float() * scale
            else:  # per-row scale
                scale = target.flatten(1).abs().amax(1) / 448.0
                q = (target / scale.reshape(-1, *([1] * (v.ndim - 1)))).to(f8)
                ft_file[k[:-6] + "weight_scale"] = scale
                ft_dense[k] = q.float() * scale.reshape(-1, *([1] * (v.ndim - 1)))
            ft_file[k] = q
        else:  # unscaled fp8 bias / norm, as in the WAN-L.O.V.E. file
            q = target.to(f8)
            ft_file[k] = q
            ft_dense[k] = q.float()
    ft_file["scaled_fp8"] = torch.zeros(2, dtype=f8)
    with _Env() as env:
        env.put("diffusion_models", "base.safetensors", base)
        env.put("diffusion_models", "ft8.safetensors", ft_file)
        with lora._TensorFile(os.path.join(env.dirs["diffusion_models"], "ft8.safetensors")) as f:
            for k in SHAPES:  # the dequant itself, every tensor
                assert torch.equal(lora._read(f, k, "ft8"), ft_dense[k]), k
        got, _ = env.run("ft8.safetensors", "base.safetensors", rank=4096)  # full rank: exact
        for k in SHAPES:
            # 2D only: extract_lora caps a conv's rank at its input channels (as the KJ original does)
            if len(SHAPES[k]) == 2:
                stem = "diffusion_model." + k.rsplit(".", 1)[0]
                up, down = got[stem + ".lora_up.weight"], got[stem + ".lora_down.weight"]
                rec = (up.flatten(1) @ down.flatten(1)).reshape(SHAPES[k])
                assert torch.allclose(rec, ft_dense[k] - base[k], atol=1e-4), k
        assert not any("scale" in k for k in got), sorted(got)


def _refused(env, ft, base, words):
    try:
        env.run(ft, base)
    except ValueError as e:
        assert words in str(e), str(e)
        return
    raise AssertionError(f"{ft} vs {base} was not refused")


def test_refusals():
    base = _base()
    with _Env() as env:
        env.put("diffusion_models", "base.safetensors", base)
        int8 = dict(base)
        int8["blocks.0.attn.weight"] = torch.zeros(16, 24, dtype=torch.int8)
        env.put("diffusion_models", "int8.safetensors", int8)
        _refused(env, "int8.safetensors", "base.safetensors", "int8")
        wide = dict(base)
        wide["blocks.0.attn.weight"] = torch.zeros(16, 32)
        env.put("diffusion_models", "wide.safetensors", wide)
        _refused(env, "wide.safetensors", "base.safetensors", "not the same model architecture")
        env.put("diffusion_models", "other.safetensors", {"x.weight": torch.zeros(2, 2)})
        _refused(env, "other.safetensors", "base.safetensors", "does not look like a fine-tune")
        open(os.path.join(env.dirs["diffusion_models"], "m.gguf"), "wb").close()
        _refused(env, "m.gguf", "base.safetensors", "only .safetensors")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all MpiLoraExtractFiles tests passed")
