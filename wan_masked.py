"""Wan 2.1 I2V conditioning from a whole guide video and its hole masks.

Ported from ComfyUI-SplatKit's WanI2VMaskedConditioning (nodes/wan.py), which reproduces
Matrix-3D's masked-video latent concat: core WanImageToVideo conditions on ONE start frame,
Matrix-3D's 360 LoRA on a FULL video whose unknown pixels are black plus a per-pixel mask. Same
maths here; only the mask's polarity changes: Cubric Studio renders its guide with the HOLES white.

Portions: MIT License, Copyright (c) 2026 mickmumpitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn.functional as F


def masked_guide(guide, holes, width, height, length):
    """The guide resized to `width` x `height`, `length` frames, its holes black; and the KNOWN mask
    [length, H, W] (1 = known). Frames past the guide's end are holes."""
    from comfy.utils import common_upscale

    vid = common_upscale(guide[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    image = torch.zeros((length, height, width, 3), dtype=vid.dtype, device=vid.device)
    image[: vid.shape[0]] = vid[..., :3]
    h = holes[:length]
    if h.ndim == 4:
        h = h[..., 0]
    h = common_upscale(h.unsqueeze(1).float(), width, height, "bilinear", "center").squeeze(1)
    known = torch.zeros((length, height, width), dtype=torch.float32, device=h.device)
    known[: h.shape[0]] = (h <= 0.5).float()
    image[known.to(image.device) < 0.5] = 0.0  # the VAE never sees what the render left behind a hole
    return image, known


def latent_mask(known, hl, wl):
    """ComfyUI's concat_mask (1 = generate) at latent size [1, 1, t, hl, wl]: area-pooled to the
    latent grid, latent frame 0 = pixel frame 0, each later latent frame = the mean of its 4 pixel
    frames, then thresholded at half (SplatKit's packing, measured on MPI-623's first path)."""
    t_lat = (known.shape[0] - 1) // 4 + 1
    mv = F.interpolate(known.unsqueeze(1), size=(hl, wl), mode="area").squeeze(1)
    groups = [mv[0:1]] + [mv[4 * i - 3: 4 * i + 1].mean(dim=0, keepdim=True) for i in range(1, t_lat)]
    return (1.0 - (torch.cat(groups) > 0.5).float()).view(1, 1, t_lat, hl, wl)


class MpiWanMaskedVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "guide": ("IMAGE", {"tooltip": "The guide video: what is known along the camera path, holes any colour."}),
                "holes": ("IMAGE", {"tooltip": "One mask a guide frame, WHITE where Wan must invent (a hole), black where the guide is known."}),
                "width": ("INT", {"default": 1440, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 720, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "MpiNodes/Wan"
    DESCRIPTION = ("Wan 2.1 image-to-video conditioning from a whole guide video plus its hole masks "
                   "(Matrix-3D's masked-video conditioning, ported from ComfyUI-SplatKit, MIT): Wan keeps "
                   "the known pixels and invents the holes.")

    def encode(self, positive, negative, vae, guide, holes, width, height, length, clip_vision_output=None):
        import comfy.model_management
        import node_helpers

        image, known = masked_guide(guide, holes, width, height, length)
        concat_latent_image = vae.encode(image)
        concat_mask = latent_mask(known, concat_latent_image.shape[-2], concat_latent_image.shape[-1])
        values = {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask}
        if clip_vision_output is not None:
            values["clip_vision_output"] = clip_vision_output
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)
        t_lat = (length - 1) // 4 + 1
        latent = torch.zeros([1, 16, t_lat, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return (positive, negative, {"samples": latent})
