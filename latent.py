import os

import torch
import safetensors.torch  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore
from comfy_execution.graph import ExecutionBlocker  # type: ignore


def _latent_path(filename: str):
    """Resolve a latent filename to an absolute .latent path.

    Absolute paths pass through; a bare name resolves against
    `<output>/latents/` — same convention as MpiHasAudio's video_path.
    Returns None for an empty name.
    """
    name = filename.strip()
    if not name:
        return None
    if not name.lower().endswith(".latent"):
        name += ".latent"
    if os.path.isabs(name):
        return name
    # basename() so a typed "../../foo" cannot escape the latents folder
    return os.path.join(
        folder_paths.get_output_directory(), "latents", os.path.basename(name)
    )


class MpiSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename": (
                    "STRING",
                    {
                        "default": "mpi_stage1",
                        "tooltip": "Saved to <output>/latents/<filename>.latent, overwriting. An absolute path is used as-is. Feed the same name to Mpi Load Latent.",
                    },
                ),
                "boolean": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "continue",
                        "label_off": "block",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("samples", "continue")
    OUTPUT_NODE = True
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Save a latent to disk, then optionally stop the branch there (block) or carry on (continue). Handles packed audio+video latents (MiniMax H3) that crash the core Save Latent node. The boolean output is not blocked, so it can still drive another branch."
    FUNCTION = "doit"

    def doit(self, samples, filename: str, boolean: bool):
        path = _latent_path(filename)
        if path is None:
            raise ValueError("Mpi Save Latent: filename is empty")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        t = samples["samples"]
        out = {"latent_format_version_0": torch.tensor([])}
        if getattr(t, "is_nested", False):
            # A NestedTensor (H3's video+audio pair) has no .contiguous(), which
            # is exactly why core SaveLatent crashes on it. unbind() hands back
            # the plain tensors; numbered keys let load rebuild the pair.
            for i, part in enumerate(t.unbind()):
                out["latent_tensor_{}".format(i)] = part.contiguous()
        else:
            # Same key core SaveLatent writes, so core LoadLatent still reads it.
            out["latent_tensor"] = t.contiguous()

        comfy.utils.save_torch_file(out, path)

        if boolean:
            return (samples, True)
        return (ExecutionBlocker(None), False)


class MpiLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (
                    "STRING",
                    {
                        "default": "mpi_stage1",
                        "tooltip": "Read from <output>/latents/<filename>.latent, or an absolute path. A plain text field, not a dropdown, so a file written this session needs no UI refresh.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("samples", "loaded")
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Load a latent written by Mpi Save Latent and continue the run. Rebuilds packed audio+video latents (MiniMax H3). If the file is missing the latent output blocks downstream execution and loaded is false, so the other branch can generate it instead."
    FUNCTION = "doit"

    def doit(self, filename: str):
        path = _latent_path(filename)
        if path is None or not os.path.isfile(path):
            return (ExecutionBlocker(None), False)

        data = safetensors.torch.load_file(path, device="cpu")
        parts = sorted(
            (k for k in data if k.startswith("latent_tensor_")),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )
        if parts:
            # Imported here, not at module scope: comfy.nested_tensor only
            # exists from ComfyUI 0.30.0, and a module-level import would stop
            # the whole pack loading on older engines.
            from comfy.nested_tensor import NestedTensor  # type: ignore

            t = NestedTensor([data[k].float() for k in parts])
        else:
            t = data["latent_tensor"].float()
        return ({"samples": t}, True)

    @classmethod
    def IS_CHANGED(cls, filename: str):
        path = _latent_path(filename)
        if path is None or not os.path.isfile(path):
            return "missing"
        # ponytail: mtime+size, not a hash — an H3 latent is hundreds of MB and
        # hashing it every graph validation would cost more than the load.
        return "{}:{}".format(os.path.getmtime(path), os.path.getsize(path))
