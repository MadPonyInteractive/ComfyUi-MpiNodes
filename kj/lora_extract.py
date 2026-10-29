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


NODE_CLASS_MAPPINGS = {
    "MpiLoraExtract": MpiLoraExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiLoraExtract": "Mpi LoRA Extract",
}
