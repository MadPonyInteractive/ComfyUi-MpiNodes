"""Bernini-R in-context conditioning, re-shaped so a host app can drive it.

Core's `BerniniConditioning` takes its references through AUTOGROW slots: the
author drags out one slot at a time and the API names them
`reference_images.reference_image_0`, `_1`, and so on. A host app that compiles a
SAVED workflow cannot grow a slot at inject time, so covering "how many
references did the user actually supply" would need one pre-authored branch per
combination. This node takes the whole surface as flat, always-present optional
inputs and drops the empty ones itself — the same trick, and for the same
reason, as `MpiH3References`.

Unlike H3 there is no prompt-tag rewriting to do. H3's prompt addresses its
references by ordinal (`<Picture 1>`), so a dropped slot has to renumber the
tags after it. Bernini's prompt cannot name a reference at all: core tells the
streams apart with a `source_id` rotation composed into the spatial RoPE, never
by text. So dropping empties here is only about not handing the model a garbage
stream — the surviving order carries no meaning a prompt can depend on.

"Empty" is not guessed from content. `MpiLoadImageFromPath` / `MpiLoadVideo`
with `block_if_empty` OFF emit a 1x1 image, which no real reference can be. A
genuinely black reference reports its true size and passes through untouched.

The conditioning work itself is DELEGATED to the core node rather than copied,
so its tensor maths moves with ComfyUI instead of drifting in a private copy.
"""

try:
    from .h3 import is_blank_image
except ImportError:  # `python bernini.py`, for the self-check at the bottom
    from h3 import is_blank_image

# Mirrors nodes.MAX_RESOLUTION, inlined to match h3.py.
MAX_RESOLUTION = 16384

# Core allows 8 autogrow slots. Four covers every reference layout the app
# offers, and each stream rides through every sampling step, so the slots are
# not free. More than four is still reachable from a graph: batch them into one
# slot and core makes a stream per image (at one shared resolution, which is the
# trade against giving each reference its own slot and its own native aspect).
MAX_REF_IMAGES = 4


def effective_length(length: int) -> int:
    """Frames Bernini will actually return for a requested `length`.

    Core builds `((length - 1) // 4) + 1` latent frames and the VAE decodes each
    back to 4, so anything off the 4n+1 grid comes back SHORT. It never says so:
    the latent is simply smaller and the shortfall only surfaces downstream,
    where a stitch node gets a different frame count than the plate it has to go
    back into. Reported here so the graph can trim the source to match.
    """
    return 4 * ((max(1, int(length)) - 1) // 4) + 1


class MpiBerniniConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "source_video": ("IMAGE", {"tooltip": "The clip being edited (v2v / rv2v). Resized to width x height with a centre cover-crop, and trimmed to length. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."}),
            "reference_video": ("IMAGE", {"tooltip": "Moving content to insert into the source video (ads2v). Scaled to ref_max_size, aspect kept."}),
        }
        for i in range(1, MAX_REF_IMAGES + 1):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": "Reference image injected as in-context tokens (r2v / rv2v), encoded at its own native aspect. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."})
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4, "tooltip": "Requested frame count. Off the 4n+1 grid it comes back SHORT - read the length output for what you will actually get."}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": MAX_RESOLUTION, "step": 16, "tooltip": "Cap for the LONG EDGE of reference_video and ref_image_*, aspect kept, snapped to 16px. Never upscales, so setting it at or above a reference's own long edge does nothing. Does NOT affect source_video, which uses width/height."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "length")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Bernini-R in-context conditioning with every reference slot exposed at "
        "once: a source video, a reference video, and 4 reference images. Empty "
        "slots are dropped and the rest renumbered, so ONE graph covers every "
        "combination instead of a branch per combination - core's own slots are "
        "autogrow, which a host app compiling a saved workflow cannot add to at "
        "inject time. An empty slot means either nothing connected or an Mpi "
        "loader with block_if_empty OFF (a 1x1 image); a real black image is not "
        "mistaken for empty. The task is inferred from what survives, exactly as "
        "core documents it: nothing = t2v, source = v2v, source + refs = rv2v, "
        "refs alone = r2v, source + reference video = ads2v. The conditioning is "
        "built by core's own BerniniConditioning. Also reports the frame count "
        "you will actually get back, which core snaps to a 4n+1 grid silently - "
        "feed it to the node that trims the plate so a stitch downstream is not "
        "handed two different frame counts."
    )
    FUNCTION = "doit"

    def doit(self, positive, negative, vae, width, height, length, ref_max_size=848, **kwargs):
        # Imported here, not at module scope: the pack must still load on a
        # ComfyUI older than 0.34, where Bernini does not exist at all.
        try:
            from comfy_extras.nodes_bernini import BerniniConditioning  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Bernini nodes are missing - MpiBerniniConditioning needs a ComfyUI "
                "with comfy_extras/nodes_bernini.py (0.34 or newer)."
            ) from exc

        # Core reads the autogrow dict with sorted(), and pairs nothing by index,
        # so the surviving references only have to be contiguous from 0.
        references = {}
        for i in range(1, MAX_REF_IMAGES + 1):
            image = kwargs.get(f"ref_image_{i}")
            if is_blank_image(image):
                continue
            references[f"reference_image_{len(references)}"] = image

        source_video = kwargs.get("source_video")
        reference_video = kwargs.get("reference_video")

        output = BerniniConditioning.execute(
            positive=positive, negative=negative, vae=vae,
            width=width, height=height, length=length, batch_size=1,
            source_video=None if is_blank_image(source_video) else source_video,
            reference_video=None if is_blank_image(reference_video) else reference_video,
            reference_images=references or None,
            ref_max_size=ref_max_size,
        )
        cond_positive, cond_negative, latent = output.result
        return (cond_positive, cond_negative, latent, effective_length(length))


if __name__ == "__main__":
    # ponytail: the arithmetic is the only part worth a check, and it needs no
    # ComfyUI. `python bernini.py` runs it.
    class _Img:
        def __init__(self, h, w):
            self.shape = (1, h, w, 3)

    assert is_blank_image(None) and is_blank_image(_Img(1, 1))
    assert not is_blank_image(_Img(1, 512)), "a 1px-tall strip is still real media"

    # on the grid: unchanged
    for n in (1, 5, 45, 49, 81):
        assert effective_length(n) == n, n
    # off the grid: reports the SHORT count core will really return
    assert effective_length(48) == 45
    assert effective_length(72) == 69
    assert effective_length(4) == 1
    # never below one frame, whatever it is handed
    assert effective_length(0) == 1 and effective_length(-10) == 1
    # what it reports is exactly what core's latent maths decodes back to
    for n in range(1, 400):
        assert effective_length(n) == 4 * (((n - 1) // 4) + 1 - 1) + 1, n
    print("bernini.py self-check OK")
