"""A preview decoder ComfyUI cannot load falls back to the colour preview (Cubric Studio MPI-1467).

ComfyUI picks a video preview decoder by FILE NAME: the first file in `models/vae_approx/` whose
name starts with the latent format's `taesd_decoder_name` (`taeh3` for MiniMax H3 since ComfyUI
0.42). It builds a `VAE` from it without checking that the file is that decoder. A different file
with the same name (Kijai's 2D MiniMax-H3-TAE, another tool's download, a broken copy) builds an
empty `VAE` ("No VAE weights detected"), and `get_previewer` then sets an attribute on its missing
model, so the sampler dies before its first step: `'NoneType' object has no attribute
'show_progress_bar'`. A user who shares a ComfyUI models folder with Cubric Studio hit this on
every H3 video.

A MISSING decoder already falls back to the colour (Latent2RGB) preview in core. This makes an
unloadable one do the same: `latent_preview.VAE` refuses a decoder it built no model for, and
`get_previewer` answers that refusal, and only that one, with the colour preview and a warning
naming the file. Every other error still raises.

No node: it applies to every sampler, core's and ours alike.
"""
import functools
import logging

_installed = False
_warned = set()


class UnusableDecoder(Exception):
    """A vae_approx file matched a decoder name but is not a decoder ComfyUI can load."""


def _checked_vae(vae_cls):
    @functools.wraps(vae_cls)
    def build(*args, **kwargs):
        vae = vae_cls(*args, **kwargs)
        if getattr(vae, "first_stage_model", None) is None:
            raise UnusableDecoder()
        return vae
    return build


def _colour_previewer(latent_preview, latent_format):
    if latent_format.latent_rgb_factors is None:
        return None
    return latent_preview.Latent2RGBPreviewer(
        latent_format.latent_rgb_factors,
        latent_format.latent_rgb_factors_bias,
        latent_format.latent_rgb_factors_reshape,
    )


def _guarded(latent_preview, get_previewer):
    @functools.wraps(get_previewer)
    def guarded(device, latent_format, *args, **kwargs):
        try:
            return get_previewer(device, latent_format, *args, **kwargs)
        except UnusableDecoder:
            name = latent_format.taesd_decoder_name
            if name not in _warned:
                _warned.add(name)
                logging.warning(
                    "[MpiNodes] models/vae_approx has a file named %s* that is not a %s preview "
                    "decoder ComfyUI can load; using the colour preview. Remove or replace that "
                    "file for sharper previews.", name, name)
            return _colour_previewer(latent_preview, latent_format)
    return guarded


def install():
    """Patch latent_preview once. Safe to call again."""
    global _installed
    if _installed:
        return
    import latent_preview  # type: ignore
    latent_preview.VAE = _checked_vae(latent_preview.VAE)
    latent_preview.get_previewer = _guarded(latent_preview, latent_preview.get_previewer)
    _installed = True
