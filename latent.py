import os

import torch
import safetensors.torch  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore
from comfy_execution.graph import ExecutionBlocker  # type: ignore


LATENT_SUBFOLDER = "latents"


def _latent_path(filename: str, search_input: bool = False):
    """Resolve a latent filename to an absolute .latent path.

    Absolute paths pass through; a bare name resolves against
    `<output>/latents/` — same convention as MpiHasAudio's video_path.
    Returns None for an empty name.

    `search_input=True` (load only) checks the engine `input/` dir FIRST and
    uses it when the file is there. That is where a host app stages a latent it
    downloaded from a previous run — ComfyUI's own LoadLatent reads `input/`
    only, so a two-stage flow driven from outside ComfyUI puts the file there.
    Falling back to `<output>/latents/` keeps a hand-run bench graph working
    unchanged: nothing stages into `input/`, so it resolves exactly as before.
    """
    name = filename.strip()
    if not name:
        return None
    if not name.lower().endswith(".latent"):
        name += ".latent"
    if os.path.isabs(name):
        return name
    # basename() so a typed "../../foo" cannot escape the latents folder
    base = os.path.basename(name)
    if search_input:
        staged = os.path.join(folder_paths.get_input_directory(), base)
        if os.path.isfile(staged):
            return staged
    return os.path.join(
        folder_paths.get_output_directory(), LATENT_SUBFOLDER, base
    )


def _save_latent_file(samples, filename: str):
    """Write a latent to `filename` and return the `ui` dict describing it.

    Shared by MpiSaveLatent and MpiStageLatents so the NestedTensor packing and the
    `ui.latents` report — which is how a host app learns what was written — have ONE
    implementation. Raises on an empty filename rather than inventing a default.
    """
    path = _latent_path(filename)
    if path is None:
        raise ValueError("Latent save: filename is empty")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    t = samples["samples"]
    out = {"latent_format_version_0": torch.tensor([])}
    if getattr(t, "is_nested", False):
        # A NestedTensor (H3's video+audio pair) has no .contiguous(), which is
        # exactly why core SaveLatent crashes on it. unbind() hands back the plain
        # tensors; numbered keys let load rebuild the pair.
        for i, part in enumerate(t.unbind()):
            out["latent_tensor_{}".format(i)] = part.contiguous()
    else:
        # Same key core SaveLatent writes, so core LoadLatent still reads it.
        out["latent_tensor"] = t.contiguous()

    comfy.utils.save_torch_file(out, path)

    # Report the saved file the way core SaveLatent does. A host app driving a
    # two-stage run reads this from /history to learn what was written — without it
    # the save is invisible outside ComfyUI and the app cannot fetch the latent to
    # continue from. `subfolder` matches where we wrote.
    return {
        "latents": [
            {
                "filename": os.path.basename(path),
                "subfolder": "" if os.path.isabs(filename.strip()) else LATENT_SUBFOLDER,
                "type": "output",
            }
        ]
    }


def _load_latent_file(path: str):
    """Read a latent written by `_save_latent_file`, rebuilding a packed pair."""
    data = safetensors.torch.load_file(path, device="cpu")
    parts = sorted(
        (k for k in data if k.startswith("latent_tensor_")),
        key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    if parts:
        # Imported here, not at module scope: comfy.nested_tensor only exists from
        # ComfyUI 0.30.0, and a module-level import would stop the whole pack
        # loading on older engines.
        from comfy.nested_tensor import NestedTensor  # type: ignore

        t = NestedTensor([data[k].float() for k in parts])
    else:
        t = data["latent_tensor"].float()
    return {"samples": t}


class MpiSaveLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"lazy": True}),
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
            "optional": {
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "enabled",
                        "label_off": "skipped",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("samples", "continue")
    OUTPUT_NODE = True
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Save a latent to disk, then optionally stop the branch there (block) or carry on (continue). Handles packed audio+video latents (MiniMax H3) that crash the core Save Latent node. The boolean output is not blocked, so it can still drive another branch. Turn `enabled` off on runs that must not save (e.g. a stage-2 continue): this is an output node, so ComfyUI runs it on EVERY submit and would otherwise drag the whole sampler feeding it along with it. `enabled` off skips that work entirely."
    FUNCTION = "doit"

    def check_lazy_status(self, filename: str, boolean: bool, samples=None, enabled=True):
        # `samples` is only needed when we are actually going to save. Without
        # this, OUTPUT_NODE guarantees execution and the sampler upstream runs
        # on every submit — ExecutionBlocker downstream cannot prevent that.
        return ["samples"] if enabled else []

    def doit(self, filename: str, boolean: bool, samples=None, enabled=True):
        if not enabled:
            return (ExecutionBlocker(None), False)
        ui = _save_latent_file(samples, filename)
        result = (samples, True) if boolean else (ExecutionBlocker(None), False)
        return {"ui": ui, "result": result}


class MpiStageLatents:
    """The whole two-stage handshake as ONE node.

    Replaces the eight-node cluster every multi-stage video workflow used to carry
    (MpiSaveLatent + MpiLoadLatent + two MpiBooleanInvert + MpiIfElse + MpiBlocker +
    MpiBooleanCompare + two MpiSimpleBoolean). `is_continue` / `is_preview` are
    WIDGETS here, not wired booleans, so a host app addresses them by
    `<title>.is_continue` instead of needing a separate titled node each.

        run            is_continue  is_preview   latent out       denoised out   saves
        full single    false        false        pass through     blocked        yes
        preview        false        true         blocked          pass through   yes
        continue       true         -            loaded from disk blocked        no

    `denoised` is the preview branch: it is released ONLY on a preview run, which is
    what stops a full run from also emitting a preview image. `latent` is the stage-2
    branch: blocked on a preview run so stage 2 does not follow the preview.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "is_continue": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "continue",
                        "label_off": "stage 1",
                        "tooltip": "Stage 2. Loads `load_path` and returns it as `latent`; nothing is saved and — because the latent inputs are lazy — the stage-1 sampler upstream never runs at all.",
                    },
                ),
                "is_preview": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "preview",
                        "label_off": "full run",
                        "tooltip": "Stop after stage 1. Saves the latent and releases `denoised` for the preview decode, blocking `latent` so stage 2 does not run. Ignored when is_continue is on.",
                    },
                ),
                "save_path": (
                    "STRING",
                    {
                        "default": "mpi_stage1",
                        "tooltip": "Where stage 1 writes. Absolute path used as-is; a bare name goes to <output>/latents/<name>.latent, overwriting.",
                    },
                ),
                "load_path": (
                    "STRING",
                    {
                        "default": "mpi_stage1",
                        "tooltip": "What a continue reads. Absolute path used as-is; a bare name is looked up in the engine <input>/ folder first (where a host app stages it), then <output>/latents/.",
                    },
                ),
            },
            "optional": {
                "latent": ("LATENT", {"lazy": True}),
                "denoised": ("LATENT", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "denoised")
    OUTPUT_NODE = True
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "The two-stage latent handshake in one node: saves stage 1, gates the preview, and loads the latent back on a continue. is_continue/is_preview are widgets, so a host app can drive both stages of ONE workflow file instead of shipping a separate _stage2 twin. Handles packed audio+video latents (MiniMax H3) that crash the core Save/Load Latent nodes."
    FUNCTION = "doit"

    def check_lazy_status(
        self, is_continue: bool, is_preview: bool, save_path: str, load_path: str,
        latent=None, denoised=None,
    ):
        # THE load-bearing line. A continue needs neither input, and refusing to ask
        # for them is what actually skips the stage-1 sampler — an ExecutionBlocker
        # travels downstream only, so gating after the sampler would still run it.
        if is_continue:
            return []
        return ["latent", "denoised"]

    def doit(
        self, is_continue: bool, is_preview: bool, save_path: str, load_path: str,
        latent=None, denoised=None,
    ):
        if is_continue:
            path = _latent_path(load_path, search_input=True)
            if path is None or not os.path.isfile(path):
                # Loud, not silent. A missing latent used to block every branch and
                # the run finished in ~0s with no output and no error, which reads as
                # a dead button rather than a missing file (MPI-452).
                raise FileNotFoundError(
                    "Mpi Stage Latents: continue requested but no latent at "
                    "{!r}".format(path or load_path)
                )
            return (_load_latent_file(path), ExecutionBlocker(None))

        if latent is None:
            raise ValueError("Mpi Stage Latents: stage 1 ran with no `latent` input")

        ui = _save_latent_file(latent, save_path)
        result = (
            (ExecutionBlocker(None), denoised)
            if is_preview
            else (latent, ExecutionBlocker(None))
        )
        return {"ui": ui, "result": result}

    @classmethod
    def IS_CHANGED(cls, is_continue: bool, load_path: str, **kwargs):
        if not is_continue:
            return float("nan")
        path = _latent_path(load_path, search_input=True)
        if path is None or not os.path.isfile(path):
            return "missing"
        return "{}:{}".format(os.path.getmtime(path), os.path.getsize(path))


class MpiLoadLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": (
                    "STRING",
                    {
                        "default": "mpi_stage1",
                        "tooltip": "Read from the engine <input>/ folder if the file is there, else <output>/latents/<filename>.latent. An absolute path is used as-is. A plain text field, not a dropdown, so a file written this session needs no UI refresh.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("samples", "loaded")
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Load a latent written by Mpi Save Latent and continue the run. Rebuilds packed audio+video latents (MiniMax H3). Looks in the engine input/ folder first (where a host app stages a latent from a previous run), then <output>/latents/. If the file is missing the latent output blocks downstream execution and loaded is false, so the other branch can generate it instead."
    FUNCTION = "doit"

    def doit(self, filename: str):
        path = _latent_path(filename, search_input=True)
        if path is None or not os.path.isfile(path):
            return (ExecutionBlocker(None), False)
        return (_load_latent_file(path), True)

    @classmethod
    def IS_CHANGED(cls, filename: str):
        path = _latent_path(filename, search_input=True)
        if path is None or not os.path.isfile(path):
            return "missing"
        # ponytail: mtime+size, not a hash — an H3 latent is hundreds of MB and
        # hashing it every graph validation would cost more than the load.
        return "{}:{}".format(os.path.getmtime(path), os.path.getsize(path))
