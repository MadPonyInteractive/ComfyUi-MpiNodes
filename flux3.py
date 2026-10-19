"""FLUX 3 Image Klein reference conditioning.

Core's ReferenceMultiLatent ("Set Reference Multi Latent") takes its references
through an AUTOGROW slot (`images.image0`, `images.image1`, ...): the author adds
one slot at a time. A host app compiling a saved workflow cannot grow a slot at
inject time, and an Mpi loader left empty still sends its 1x1 sentinel down the
link, which core would encode as a real (and very wrong) reference. So
`MpiFlux3References` takes all ten slots as flat, always-present optional inputs
and drops the empty ones itself - the same contract as `MpiH3References`.

Empty is not guessed from content: nothing connected, or an Mpi loader with
`block_if_empty` OFF emitting its 1x1 image (`is_blank_image`, shared with h3.py).
A genuinely black reference reports its true size and passes through.

The prompt cites references the way FLUX 3 was trained to read them, `<ref_image_0>`
for the first image it is given. Slots are numbered from 0 to match, and the prompt
is written against SLOT numbers; once empties are dropped the survivors are
renumbered, so the prompt is rewritten here, where both numberings are known.

The conditioning itself is DELEGATED to core's ReferenceMultiLatent rather than
copied: its tensor maths moves with ComfyUI and a private copy would drift.

Deliberately free of ComfyUI imports at module scope so `python flux3.py` runs the
self-check at the bottom in bare Python.
"""

import re

try:
    from .h3 import is_blank_image
except ImportError:  # `python flux3.py`, no package
    from h3 import is_blank_image

MAX_REF_IMAGES = 10

# BFL's own prompts cite a reference two ways: `<ref_image_0>` in the prose and a bare
# `"from": "ref_image_0"` in the i2i layout table. Both are rewritten.
_TAG_RE = re.compile(r"(<?)\bref_image_(\d+)\b(>?)")


def collect_refs(slots: dict):
    """Drop the empty `ref_image_0..9` slots, renumber the survivors from 0.

    Returns `images` in core's Autogrow shape (`{"image0": ..., "image1": ...}`)
    and `kept`, the ORIGINAL slot numbers that survived, in order.
    """
    images, kept = {}, []
    for i in range(MAX_REF_IMAGES):
        image = slots.get(f"ref_image_{i}")
        if is_blank_image(image):
            continue
        images[f"image{len(images)}"] = image
        kept.append(i)
    return images, kept


def rewrite_prompt_tags(prompt: str, kept) -> str:
    """Translate slot-numbered `ref_image_N` into the index core presents.

    A tag naming an EMPTY slot is dropped when bracketed (`<ref_image_3>`): the
    model would be told to look for a reference that is not there. A bare
    `ref_image_3` (a layout-table `from` value) is left alone - removing it would
    break the JSON, and a row about a missing image is wrong either way.
    """
    mapping = {slot: index for index, slot in enumerate(kept)}

    def _sub(match):
        opening, slot, closing = match.group(1), int(match.group(2)), match.group(3)
        target = mapping.get(slot)
        if target is None:
            return "" if opening and closing else match.group(0)
        return f"{opening}ref_image_{target}{closing}"

    return re.sub(r"[ \t]{2,}", " ", _TAG_RE.sub(_sub, prompt)).strip()


def ref_tag_map(kept) -> str:
    """Which tag each surviving slot became - the `ref_tags` output."""
    if not kept:
        return "(no references - this is a plain text-to-image run)"
    return "\n".join(f"<ref_image_{index}> = ref_image_{slot}" for index, slot in enumerate(kept))


class MpiFlux3References:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "negative": ("CONDITIONING", {"tooltip": "Optional. Gets the same reference latents. Unconnected, the positive is returned in its place, as core does."}),
        }
        for i in range(MAX_REF_IMAGES):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": f"Reference image, cited in the prompt as <ref_image_{i}>. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."})
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Cite references by SLOT: <ref_image_0> is whatever is wired into ref_image_0. The node rewrites tags to the order core presents once empty slots are dropped (see ref_tags); a tag naming an empty slot is removed."}),
                "resolution": (["resize", "keep resolution"], {"default": "resize", "tooltip": "resize scales each reference to `megapixels`, aspect kept. keep resolution encodes it as it is."}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01, "tooltip": "Target size of each reference when resolution is resize. 1.0 is about 1024 x 1024."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "empty_latent", "ref_tags")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "FLUX 3 Image Klein reference conditioning with all 10 reference slots exposed at once. "
        "Empty slots are dropped and the rest renumbered, so ONE graph covers every combination "
        "instead of growing core's Autogrow input per run. Empty means nothing connected or an Mpi "
        "loader with block_if_empty OFF (a 1x1 image) - a real black image passes through. Encodes "
        "the prompt with `clip`, then core's own ReferenceMultiLatent builds the conditioning. "
        "Write the prompt against SLOT numbers - <ref_image_0> means ref_image_0 - and the node "
        "rewrites them to what core presents, dropping any tag whose slot is empty. `empty_latent` "
        "is core's: the first reference's latent times 0, or core's square 4-channel SD latent when there are none - wrong for FLUX 3, so a text-to-image run takes its latent from EmptyFlux2LatentImage instead."
    )
    FUNCTION = "doit"

    def doit(self, clip, vae, prompt, resolution="resize", megapixels=1.0, negative=None, **kwargs):
        # Imported here, not at module scope: the pack must still load on a ComfyUI
        # without FLUX 3 support.
        try:
            from comfy_extras.nodes_edit_model import ReferenceMultiLatent  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ReferenceMultiLatent is missing - MpiFlux3References needs ComfyUI 0.41.0 or newer."
            ) from exc

        images, kept = collect_refs(kwargs)
        prompt = rewrite_prompt_tags(prompt, kept)
        tokens = clip.tokenize(prompt)
        positive = clip.encode_from_tokens_scheduled(tokens)

        output = ReferenceMultiLatent.execute(
            positive=positive,
            negative=negative,
            vae=vae,
            resolution={"resolution": resolution, "megapixels": megapixels},
            images=images or None,
        )
        pos, neg, empty_latent = output.result
        return (pos, neg, empty_latent, ref_tag_map(kept))


if __name__ == "__main__":
    class _Img:
        def __init__(self, h, w):
            self.shape = (1, h, w, 3)

    real, blank = _Img(512, 768), _Img(1, 1)
    images, kept = collect_refs({"ref_image_0": blank, "ref_image_2": real, "ref_image_5": real, "ref_image_9": None})
    assert kept == [2, 5], kept
    assert list(images) == ["image0", "image1"], images
    assert images["image0"] is real

    p = rewrite_prompt_tags("Put the hat from <ref_image_5> on the man in <ref_image_2>, ignore <ref_image_0>.", kept)
    assert p == "Put the hat from <ref_image_1> on the man in <ref_image_0>, ignore .", p
    # Bare layout-table values move with the brackets; an empty one is left alone.
    assert rewrite_prompt_tags('{"from": "ref_image_5"} {"from": "ref_image_7"}', kept) == '{"from": "ref_image_1"} {"from": "ref_image_7"}'
    # Not a tag: a longer word is never touched.
    assert rewrite_prompt_tags("my_ref_image_2x", kept) == "my_ref_image_2x"
    # Nothing empty: the prompt comes back as written.
    all_kept = list(range(10))
    assert rewrite_prompt_tags("<ref_image_9> and <ref_image_0>", all_kept) == "<ref_image_9> and <ref_image_0>"
    assert ref_tag_map(kept) == "<ref_image_0> = ref_image_2\n<ref_image_1> = ref_image_5"
    assert ref_tag_map([]).startswith("(no references")
    print("flux3.py self-check passed")
