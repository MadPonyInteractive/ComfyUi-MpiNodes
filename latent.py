import os

import torch
import safetensors.torch  # type: ignore
import comfy.utils  # type: ignore
import folder_paths  # type: ignore
from comfy_execution.graph import ExecutionBlocker  # type: ignore

from .help_funcs import resolve_in_comfy_dir, list_input_files


LATENT_SUBFOLDER = "latents"


def _latent_path(filename: str, search_input: bool = False):
    """Resolve a latent filename to an absolute .latent path.

    An absolute path is honoured only inside ComfyUI's output/ or input/
    folders; a bare name resolves against `<output>/latents/`.
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
        # Only inside ComfyUI's own folders: an absolute path from /prompt is an
        # arbitrary file read/write under the registry's policy. Anywhere else
        # falls through to the basename rule.
        inside = resolve_in_comfy_dir(name, ("output", "input"))
        if inside:
            return inside
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
                        "tooltip": "Read from the engine <input>/ folder if the file is there, else <output>/latents/<filename>.latent. An absolute path must be inside input/ or output/. A plain text field, so a file written this session needs no UI refresh. Clear it to use the file picker.",
                    },
                ),
            },
            "optional": {
                "file": (
                    list_input_files((".latent",)),
                    {"tooltip": "A .latent file in ComfyUI's input/ folder. Used only when filename is empty."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, file=None):
        return True  # a missing file blocks downstream instead of erroring

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("samples", "loaded")
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Load a latent written by Mpi Save Latent and continue the run. Rebuilds packed audio+video latents (MiniMax H3). Looks in the engine input/ folder first (where a host app stages a latent from a previous run), then <output>/latents/. If the file is missing the latent output blocks downstream execution and loaded is false, so the other branch can generate it instead."
    FUNCTION = "doit"

    def doit(self, filename: str, file=None):
        path = _latent_path(filename.strip() or file or "", search_input=True)
        if path is None or not os.path.isfile(path):
            return (ExecutionBlocker(None), False)
        return (_load_latent_file(path), True)

    @classmethod
    def IS_CHANGED(cls, filename: str, file=None):
        path = _latent_path(filename.strip() or file or "", search_input=True)
        if path is None or not os.path.isfile(path):
            return "missing"
        # ponytail: mtime+size, not a hash — an H3 latent is hundreds of MB and
        # hashing it every graph validation would cost more than the load.
        return "{}:{}".format(os.path.getmtime(path), os.path.getsize(path))


class MpiLatentUpscale:
    """Latent upscale that survives a packed audio+video latent (MiniMax H3).

    Core's Upscale Latent gets two things wrong here, and both are silent until
    they are not:

    1. It hands the latent straight to `comfy.utils.common_upscale`, which
       `.reshape()`s it. A NestedTensor has no `reshape`, so an H3 latent dies with
       `AttributeError: 'NestedTensor' object has no attribute 'reshape'` - the same
       family of failure that made MpiSaveLatent necessary.
    2. It converts pixels to latent cells with a hardcoded `// 8`, the SD VAE
       factor. H3's video latent is `/16`, so a target typed in pixels comes out at
       DOUBLE the size with no error at all: type 1344x768 and get 2688x1536.

    So this one takes the target in PIXELS and divides by `stride`, and it upscales
    only the half of the pair that has spatial dims. The audio half is
    `[B, 32, 2, T]` - time, no height or width - so it rides through untouched and
    the pair stays aligned for the second sampler.

    The point of it is the hi-res fix: sample stage 1 on a small canvas, upscale the
    latent here, and let stage 2's low sigmas finish at the big one, so only the
    second half of the denoise pays for the resolution.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_method": (cls.upscale_methods, {"default": "nearest-exact"}),
                "width": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Target width in PIXELS, not latent cells. Unlike core's node this is the number you actually want.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Target height in PIXELS.",
                    },
                ),
                "stride": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Pixels per latent cell for THIS model's VAE. 16 for MiniMax H3 and Krea2; 8 for the SD/SDXL family. Core hardcodes 8, which is why its node doubles an H3 target.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Upscale a latent to a target given in PIXELS, dividing by the model's own VAE stride instead of core's hardcoded 8. Handles the packed audio+video latents (MiniMax H3) that crash core's Upscale Latent, scaling only the video half and passing the audio half through. For a hi-res fix: small stage 1, upscale here, finish stage 2 at the target."
    FUNCTION = "doit"

    def doit(self, samples, upscale_method: str, width: int, height: int, stride: int):
        lw = max(1, width // stride)
        lh = max(1, height // stride)

        t = samples["samples"]
        if getattr(t, "is_nested", False):
            # Imported here for the same reason _load_latent_file does it: the
            # module only exists from ComfyUI 0.30.0 and a module-level import
            # would stop the whole pack loading on an older engine.
            from comfy.nested_tensor import NestedTensor  # type: ignore

            parts = list(t.unbind())
            # Pick the video half by SHAPE, not by index. The pair is video-then-audio
            # today, but indexing on that would break silently if it ever flipped,
            # and "has spatial dims" is the actual property being relied on.
            for i, part in enumerate(parts):
                if part.dim() >= 5:
                    parts[i] = comfy.utils.common_upscale(
                        part, lw, lh, upscale_method, "disabled"
                    )
            out = NestedTensor(parts)
        else:
            out = comfy.utils.common_upscale(t, lw, lh, upscale_method, "disabled")

        s = samples.copy()
        s["samples"] = out
        return (s,)


class MpiH3WindowJoin:
    """Join a windowed MiniMax H3 run in LATENT space, before its one decode (Cubric Studio MPI-1608 Phase 3).

    The last window of a windowed run (Video Edit) wires its own latent in and names the earlier windows' saved
    latents in `windows`; the output is the whole film as one AV latent, so the graph's own refine, decode and
    save make the film in one go. `windows` empty = a pass-through, so every other window (and a clip that fits
    one window) runs exactly as before.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "This window's AV latent (the last window of the run)."}),
                "windows": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "The EARLIER windows' saved latents, oldest first, comma or newline separated - the names Mpi Save Latent wrote them under (looked up in input/ first, then <output>/latents/). Empty = pass this window through untouched.",
                    },
                ),
                "context_frames": (
                    "INT",
                    {"default": 39, "min": 39, "max": 1000, "step": 51,
                     "tooltip": "The head each window held from the one before (Mpi H3 Masked Prefix's context_frames). Every head is checked against the tail it copied: a window from another run raises instead of joining with a jump."},
                ),
                "lead_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000,
                     "tooltip": "Frames at the start of the FIRST window that are not part of the film (a lead-in), dropped. Must end on a latent step (17 = 5 steps). Ignored when `windows` is empty."},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING")
    RETURN_NAMES = ("latent", "added_frames", "report")
    CATEGORY = "MpiNodes/Latent"
    DESCRIPTION = "Join the windows of a windowed MiniMax H3 run in latent space: the earlier windows (by saved name) + this one, each later window's held head checked and dropped, so ONE decode makes the whole film with no pixel seams. added_frames = the frames placed in front of this window (add it to the frame count a trim keeps). Empty `windows` passes the latent through."
    FUNCTION = "doit"

    def doit(self, latent, windows: str, context_frames: int, lead_frames: int = 0):
        names = [n.strip() for n in windows.replace("\n", ",").split(",") if n.strip()]
        if not names:
            return (latent, 0, "one window: passed through")
        from comfy.nested_tensor import NestedTensor  # type: ignore
        from .h3 import _unpack_av, latent_frames
        from .h3_window_join import join_windows

        pieces = []
        for name in names:
            path = _latent_path(name, search_input=True)
            if path is None or not os.path.isfile(path):
                raise FileNotFoundError(
                    "Mpi H3 Window Join: no saved window latent {!r} (looked for {})".format(name, path))
            pieces.append(_unpack_av(_load_latent_file(path), name))
        current = _unpack_av(latent, "latent")
        pieces.append(current)
        v, a, report = join_windows(pieces, context_frames, lead_frames)
        added = latent_frames(v.shape[2]) - latent_frames(current[0].shape[2])
        print("[MpiH3WindowJoin]", report.replace("\n", " | "))
        return ({"samples": NestedTensor([v, a])}, added, report)
