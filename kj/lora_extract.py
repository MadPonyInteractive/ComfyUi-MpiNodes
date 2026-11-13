"""Extract a LoRA from the difference of two models or two CLIPs (MPI-1341).

Ported from comfyui-kjnodes (nodes/lora_nodes.py: LoraExtractKJ, extract_lora, calc_lora_model),
GPL-3.0, Copyright (c) kijai. See LICENSE-KJNodes. Input names and accepted values are the
original's. Differences, all deliberate:

  * A key whose SVD fails raises instead of being logged and left out. The original skipped it,
    which writes a LoRA with a silent hole in it.
  * torch.linalg.svd runs with full_matrices=False. Only the leading `rank` columns are used, so the
    result is the same and the huge square U is never built.
  * No tqdm bar; ComfyUI's own ProgressBar reports progress.

The output path goes through folder_paths.get_save_image_path, which refuses a prefix that leaves
ComfyUI's output folder, exactly as the core SaveImage node does.
"""
import logging
import os

import torch

import comfy.model_detection  # type: ignore
import comfy.model_management  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore

CLAMP_QUANTILE = 0.99


def extract_lora(diff, key, rank, algorithm, lora_type, lowrank_iters=7, adaptive_param=1.0, clamp_quantile=True):
    """(U, Vh) such that U @ Vh approximates the weight difference `diff`, at `rank` or at the rank
    the adaptive `lora_type` picks (capped by `rank`). Conv weights come back as 1x1 up / kxk down."""
    conv2d = (len(diff.shape) == 4)
    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    out_dim, in_dim = diff.size()[0:2]

    if conv2d:
        if conv2d_3x3:
            diff = diff.flatten(start_dim=1)
        else:
            diff = diff.squeeze()

    diff_float = diff.float()
    if algorithm == "svd_lowrank":
        lora_rank = min(rank, in_dim, out_dim)
        U, S, V = torch.svd_lowrank(diff_float, q=lora_rank, niter=lowrank_iters)
        U = U @ torch.diag(S)
        Vh = V.t()
    else:
        U, S, Vh = torch.linalg.svd(diff_float, full_matrices=False)
        # flexible rank selection like LyCORIS extract_locon
        if "adaptive" in lora_type:
            if lora_type == "adaptive_ratio":
                min_s = torch.max(S) * adaptive_param
                lora_rank = torch.sum(S > min_s).item()
            elif lora_type == "adaptive_energy":
                energy = torch.cumsum(S ** 2, dim=0)
                total_energy = torch.sum(S ** 2)
                lora_rank = torch.sum(energy < adaptive_param * total_energy).item() + 1
            elif lora_type == "adaptive_quantile":
                s_cum = torch.cumsum(S, dim=0)
                lora_rank = torch.sum(s_cum < adaptive_param * torch.sum(S)).item()
            elif lora_type == "adaptive_fro":
                S_squared = S.pow(2)
                sum_S_squared = torch.cumsum(S_squared, dim=0) / float(torch.sum(S_squared))
                lora_rank = int(torch.searchsorted(sum_S_squared, adaptive_param ** 2)) + 1
                lora_rank = max(1, min(lora_rank, len(S)))
            else:
                raise ValueError(f"Unknown adaptive lora_type {lora_type!r}")

            lora_rank = min(lora_rank, rank)  # capped by the rank setting

            if lora_type == "adaptive_fro":
                S_squared = S.pow(2)
                fro_percent = float(torch.sqrt(torch.sum(S_squared[:lora_rank])) / torch.sqrt(torch.sum(S_squared)))
                logging.info(f"{key} Extracted LoRA rank: {lora_rank}, Frobenius retained: {fro_percent:.1%}")
            else:
                logging.info(f"{key} Extracted LoRA rank: {lora_rank}")
        else:
            lora_rank = rank

        lora_rank = max(1, lora_rank)
        lora_rank = min(out_dim, in_dim, lora_rank)

        U = U[:, :lora_rank]
        S = S[:lora_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:lora_rank, :]

    if clamp_quantile:
        dist = torch.cat([U.flatten(), Vh.flatten()])
        if dist.numel() > 100_000:
            # estimate the quantile from a 100,000 element sample
            idx = torch.randperm(dist.numel(), device=dist.device)[:100_000]
            hi_val = torch.quantile(dist[idx], CLAMP_QUANTILE)
        else:
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
        U = U.clamp(-hi_val, hi_val)
        Vh = Vh.clamp(-hi_val, hi_val)
    if conv2d:
        U = U.reshape(out_dim, lora_rank, 1, 1)
        Vh = Vh.reshape(lora_rank, in_dim, kernel_size[0], kernel_size[1])
    return (U, Vh)


def calc_lora_model(model_diff, rank, prefix_model, prefix_lora, output_sd, lora_type, algorithm, lowrank_iters,
                    out_dtype, bias_diff=False, adaptive_param=1.0, clamp_quantile=True):
    """Walks the patched model's weights one at a time (low peak memory) and fills `output_sd`."""
    device = comfy.model_management.get_torch_device()
    sd_keys = []
    for name, _ in model_diff.model.named_parameters():
        if prefix_model is None or name.startswith(prefix_model):
            sd_keys.append(name)
    for name, _ in model_diff.model.named_buffers():
        if prefix_model is None or name.startswith(prefix_model):
            sd_keys.append(name)

    total_keys = len([k for k in sd_keys if k.endswith(".weight") or (bias_diff and k.endswith(".bias"))])
    comfy_pbar = comfy.utils.ProgressBar(total_keys)

    for k in sd_keys:
        if k.endswith(".weight"):
            weight_diff = model_diff.patch_weight_to_device(k, return_weight=True)
            if weight_diff is None:
                comfy_pbar.update(1)
                continue
            if weight_diff.ndim == 5:
                logging.info(f"Skipping 5D tensor for key {k}")
                del weight_diff
                comfy_pbar.update(1)
                continue
            name = k[len(prefix_model):-7]
            if lora_type != "full":
                if weight_diff.ndim < 2:
                    if bias_diff:
                        output_sd["{}{}.diff".format(prefix_lora, name)] = weight_diff.contiguous().to(out_dtype).cpu()
                    del weight_diff
                    comfy_pbar.update(1)
                    continue
                out = extract_lora(weight_diff.to(device), k, rank, algorithm, lora_type, lowrank_iters=lowrank_iters,
                                   adaptive_param=adaptive_param, clamp_quantile=clamp_quantile)
                output_sd["{}{}.lora_up.weight".format(prefix_lora, name)] = out[0].contiguous().to(out_dtype).cpu()
                output_sd["{}{}.lora_down.weight".format(prefix_lora, name)] = out[1].contiguous().to(out_dtype).cpu()
            else:
                output_sd["{}{}.diff".format(prefix_lora, name)] = weight_diff.contiguous().to(out_dtype).cpu()
            del weight_diff
            comfy_pbar.update(1)

        elif bias_diff and k.endswith(".bias"):
            weight = model_diff.patch_weight_to_device(k, return_weight=True)
            if weight is not None:
                output_sd["{}{}.diff_b".format(prefix_lora, k[len(prefix_model):-5])] = weight.contiguous().to(out_dtype).cpu()
                del weight
            comfy_pbar.update(1)

    del model_diff
    comfy.model_management.soft_empty_cache()
    return output_sd


class MpiLoraExtract:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "finetuned": ("MODEL,CLIP", {"tooltip": "The finetuned model or clip to extract LoRA from."}),
                "original": ("MODEL,CLIP", {"tooltip": "The original base model or clip to diff against."}),
                "filename_prefix": ("STRING", {"default": "loras/ComfyUI_extracted_lora"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1,
                                 "tooltip": "The rank for a standard LoRA, or the maximum rank for adaptive methods."}),
                "lora_type": (["standard", "full", "adaptive_ratio", "adaptive_quantile", "adaptive_energy", "adaptive_fro"],),
                "algorithm": (["svd_linalg", "svd_lowrank"], {
                    "default": "svd_lowrank",
                    "tooltip": "SVD algorithm; svd_lowrank is faster but less accurate."}),
                "lowrank_iters": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1,
                                          "tooltip": "Subspace iterations for the lowrank SVD algorithm."}),
                "output_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "bias_diff": ("BOOLEAN", {"default": True}),
                "adaptive_param": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "ratio mode: fraction of the largest singular value. quantile mode: the quantile of the singular values. fro mode: Frobenius norm retention."}),
                "clamp_quantile": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "MpiNodes/KJ"
    RETURN_TYPES = ()
    FUNCTION = "extract"
    OUTPUT_NODE = True
    DESCRIPTION = "Extracts a LoRA from the difference between a finetuned and an original model (or CLIP) and saves it under output/."

    def extract(self, finetuned, original, filename_prefix, rank, lora_type, algorithm, lowrank_iters,
                output_dtype, bias_diff, adaptive_param, clamp_quantile):
        if algorithm == "svd_lowrank" and lora_type != "standard":
            raise ValueError("svd_lowrank algorithm is only supported for standard LoRA extraction.")

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[output_dtype]

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)

        output_sd = {}
        is_clip = hasattr(finetuned, "patcher")

        if is_clip:
            clip_diff = finetuned.clone()
            kp = original.get_key_patches()
            kp = {k: v for k, v in kp.items() if not k.endswith(".position_ids") and not k.endswith(".logit_scale")}
            clip_diff.add_patches(kp, -1.0, 1.0)
            output_sd = calc_lora_model(clip_diff.patcher, rank, "", "text_encoders.", output_sd, lora_type, algorithm,
                                        lowrank_iters, dtype, bias_diff=bias_diff, adaptive_param=adaptive_param,
                                        clamp_quantile=clamp_quantile)
        else:
            m = finetuned.clone()
            kp = original.get_key_patches("diffusion_model.")
            m.add_patches(kp, -1.0, 1.0)
            output_sd = calc_lora_model(m, rank, "diffusion_model.", "diffusion_model.", output_sd, lora_type, algorithm,
                                        lowrank_iters, dtype, bias_diff=bias_diff, adaptive_param=adaptive_param,
                                        clamp_quantile=clamp_quantile)

        if "adaptive" in lora_type:
            rank_str = f"{lora_type}_{adaptive_param:.2f}"
        else:
            rank_str = rank
        output_checkpoint = f"{filename}_rank_{rank_str}_{output_dtype}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)

        comfy.utils.save_torch_file(output_sd, output_checkpoint, metadata=None)
        return {}


# ---------------------------------------------------------------------------------------------
# MpiLoraExtractFiles - our own (Cubric Studio MPI-1452), not part of the KJNodes port.
#
# The same extract, read straight from the two files one tensor at a time, so the peak memory is
# one layer plus the LoRA being written (a few hundred MB) instead of both models at full size
# (about 20 GB for Wan 2.2 5B in fp16). Works with no GPU. A scaled-fp8 fine-tune (ComfyUI's
# "scaled_fp8": an F8 weight + its `scale_weight`) is dequantized per tensor before the diff; the
# diff itself is float32, so it is never rounded to the fine-tune's fp8 the way a patched-model
# diff is. What it does not see: renames ComfyUI applies at load time (diffusers layouts, Flux
# norm `.scale`), so a new model family needs its LoRA keys checked against the loaded model.
# ---------------------------------------------------------------------------------------------

# Where a file keeps its diffusion model. Bare = a diffusion-only file (UNETLoader's folder).
_DIFFUSION_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# Keys of an all-in-one checkpoint that are not the diffusion model.
_NOT_DIFFUSION = ("first_stage_model.", "cond_stage_model.", "conditioner.", "text_encoders.", "vae.",
                  "model_ema.", "embedding_manager.", "lora_te", "clip_")
# Quantization bookkeeping, never a weight of its own.
_QUANT_SUFFIXES = ("scale_weight", "scale_input", "weight_scale", "input_scale", "scaled_fp8", "comfy_quant")
_FP8 = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
             if hasattr(torch, n))
_REFUSED = (torch.int8, torch.uint8, torch.int16, torch.int32)


_ST_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
              "BOOL": torch.bool}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _attr):
        _ST_DTYPES[_name] = getattr(torch, _attr)


class _TensorFile:
    """A .safetensors file read one tensor at a time with plain reads.

    Not safetensors' safe_open: that memory-maps the WHOLE file, and on Windows a copy-on-write map
    is charged to the commit limit at once (12 + 9 GB for the Wan 5B pair, measured), which is the
    "paging file is too small" failure on a small-RAM PC. Here only the tensor being read is held.
    """

    def __init__(self, path):
        import json  # stdlib: ComfyUI has it in sys.modules long before the pack's json.py loads
        self._f = open(path, "rb")
        size = int.from_bytes(self._f.read(8), "little")
        header = json.loads(self._f.read(size))
        header.pop("__metadata__", None)
        self._start = 8 + size
        self._header = header

    def keys(self):
        return list(self._header)

    def __contains__(self, key):
        return key in self._header

    def get_tensor(self, key):
        info = self._header[key]
        if info["dtype"] not in _ST_DTYPES:
            raise ValueError(f"{key}: tensor type {info['dtype']} is not supported.")
        begin, end = info["data_offsets"]
        self._f.seek(self._start + begin)
        data = bytearray(self._f.read(end - begin))
        dtype = _ST_DTYPES[info["dtype"]]
        if not data:
            return torch.empty(info["shape"], dtype=dtype)
        return torch.frombuffer(data, dtype=dtype).reshape(info["shape"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()


def _model_files():
    """Every model file the two folders hold, diffusion-only first, each name once."""
    names = []
    for folder in ("diffusion_models", "checkpoints"):
        for n in folder_paths.get_filename_list(folder):
            if n not in names:
                names.append(n)
    return sorted(names)


def _full_path(name):
    for folder in ("diffusion_models", "checkpoints"):
        path = folder_paths.get_full_path(folder, name)
        if path:
            return path
    raise ValueError(f"{name}: not in the diffusion_models or checkpoints folder.")


def _diffusion_keys(keys):
    """{bare key: file key} for the diffusion model inside a file, whatever prefix it was saved with."""
    keys = list(keys)
    # ComfyUI's own choice first (model.diffusion_model. / model.model. for audio / net.), so the
    # LoRA keys are the names the loaded model has. Its "model." is a no-match fallback, not a pick.
    picked = comfy.model_detection.unet_prefix_from_state_dict(dict.fromkeys(keys))
    for prefix in ((picked,) if picked != "model." else ()) + _DIFFUSION_PREFIXES:
        hit = {k[len(prefix):]: k for k in keys if k.startswith(prefix)}
        if hit:
            break
    else:
        hit = {k: k for k in keys if not k.startswith(_NOT_DIFFUSION)}
    return {bare: k for bare, k in hit.items() if not bare.endswith(_QUANT_SUFFIXES)}


def _read(f, key, label):
    """One tensor as float32, a scaled-fp8 weight multiplied by its scale. Integer quants refused."""
    t = f.get_tensor(key)
    if t.dtype in _REFUSED:
        raise ValueError(f"{label} is int8/integer-quantized ({key}); it cannot be extracted from. "
                         "Use the FP16, BF16 or FP8 version of this model.")
    if t.dtype in _FP8:
        t = t.to(torch.float32)
        if key.endswith("weight"):
            stem = key[:-len("weight")]
            for scale_key in (stem + "scale_weight", stem + "weight_scale"):
                if scale_key in f:
                    scale = f.get_tensor(scale_key).to(torch.float32)
                    if scale.numel() == 1:
                        t = t * scale
                    elif scale.numel() == t.shape[0]:
                        t = t * scale.reshape(-1, *([1] * (t.ndim - 1)))
                    else:
                        raise ValueError(f"{label}: {scale_key} is a block-wise scale, which is not supported.")
                    break
    return t.to(torch.float32)


class MpiLoraExtractFiles:
    @classmethod
    def INPUT_TYPES(cls):
        files = _model_files()
        return {
            "required": {
                "finetuned_name": (files, {"tooltip": "The fine-tuned model (diffusion_models or checkpoints). FP16, BF16 or scaled FP8."}),
                "original_name": (files, {"tooltip": "The original base model it was fine-tuned from."}),
                "filename_prefix": ("STRING", {"default": "loras/extracted_lora"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "algorithm": (["svd_linalg", "svd_lowrank"], {
                    "default": "svd_lowrank",
                    "tooltip": "SVD algorithm; svd_lowrank is faster but less accurate."}),
                "lowrank_iters": ("INT", {"default": 7, "min": 1, "max": 100, "step": 1}),
                "output_dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "bias_diff": ("BOOLEAN", {"default": True}),
                "clamp_quantile": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "MpiNodes/KJ"
    RETURN_TYPES = ()
    FUNCTION = "extract"
    OUTPUT_NODE = True
    DESCRIPTION = ("Extracts a LoRA from two model FILES one layer at a time (low memory, no GPU needed) "
                   "and saves it under output/. A scaled-FP8 fine-tune is dequantized first.")

    def extract(self, finetuned_name, original_name, filename_prefix, rank, algorithm, lowrank_iters,
                output_dtype, bias_diff, clamp_quantile):
        for name in (finetuned_name, original_name):
            if not name.lower().endswith((".safetensors", ".sft")):
                raise ValueError(f"{name}: only .safetensors files can be extracted from.")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[output_dtype]
        device = comfy.model_management.get_torch_device()

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)

        output_sd = {}
        with _TensorFile(_full_path(finetuned_name)) as ft, _TensorFile(_full_path(original_name)) as base:
            ft_keys, base_keys = _diffusion_keys(ft.keys()), _diffusion_keys(base.keys())
            weights = [k for k in base_keys if k.endswith((".weight", ".bias"))]
            shared = [k for k in weights if k in ft_keys]
            if not weights or len(shared) < 0.9 * len(weights):
                raise ValueError(
                    f"{finetuned_name} does not look like a fine-tune of {original_name}: "
                    f"{len(shared)} of the base's {len(weights)} weights are in both files.")

            todo = [k for k in shared if k.endswith(".weight") or bias_diff]
            pbar = comfy.utils.ProgressBar(len(todo))
            for k in todo:
                comfy.model_management.throw_exception_if_processing_interrupted()
                w_ft = _read(ft, ft_keys[k], finetuned_name)
                w_base = _read(base, base_keys[k], original_name)
                if w_ft.shape != w_base.shape:
                    raise ValueError(f"{k}: {list(w_ft.shape)} in the fine-tune, {list(w_base.shape)} in the base. "
                                     "These are not the same model architecture.")
                diff = w_ft.to(device) - w_base.to(device)
                del w_ft, w_base
                if k.endswith(".bias"):
                    output_sd[f"diffusion_model.{k[:-5]}.diff_b"] = diff.contiguous().to(dtype).cpu()
                elif diff.ndim == 5:
                    logging.info(f"Skipping 5D tensor for key {k}")
                elif diff.ndim < 2:
                    if bias_diff:
                        output_sd[f"diffusion_model.{k[:-7]}.diff"] = diff.contiguous().to(dtype).cpu()
                else:
                    up, down = extract_lora(diff, k, rank, algorithm, "standard", lowrank_iters=lowrank_iters,
                                            clamp_quantile=clamp_quantile)
                    output_sd[f"diffusion_model.{k[:-7]}.lora_up.weight"] = up.contiguous().to(dtype).cpu()
                    output_sd[f"diffusion_model.{k[:-7]}.lora_down.weight"] = down.contiguous().to(dtype).cpu()
                del diff
                pbar.update(1)

        file = f"{filename}_rank_{rank}_{output_dtype}_{counter:05}_.safetensors"
        comfy.utils.save_torch_file(output_sd, os.path.join(full_output_folder, file), metadata=None)
        comfy.model_management.soft_empty_cache()
        return {"ui": {"lora": [{"filename": file, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "MpiLoraExtract": MpiLoraExtract,
    "MpiLoraExtractFiles": MpiLoraExtractFiles,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiLoraExtract": "Mpi LoRA Extract",
    "MpiLoraExtractFiles": "Mpi LoRA Extract (Files, Low Memory)",
}
