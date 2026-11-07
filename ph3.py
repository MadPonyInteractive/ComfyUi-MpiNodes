"""P-H3 Image (PrunaAI, on MiniMax H3) conditioning.

Core's PH3ImageConditioning takes its pictures through an AUTOGROW slot
(`images.image_1`, `images.image_2`, ...): the author adds one slot at a time. A host
app compiling a saved workflow cannot grow a slot at inject time, and an Mpi loader
left empty still sends its 1x1 sentinel down the link, which core would encode as a
real picture. So `MpiPH3ImageConditioning` takes all eight slots as flat,
always-present optional inputs and drops the empty ones itself - the same contract
as `MpiH3References` and `MpiFlux3References`.

image_1 is the CANVAS: the picture being edited, whose size and aspect the output
keeps. image_2..8 are references (Pruna: up to 7 after the canvas). If image_1 is
empty, the first surviving slot becomes the canvas - that is what core would do with
the pictures it is given.

Empty is not guessed from content: nothing connected, or an Mpi loader with
`block_if_empty` OFF emitting its 1x1 image (`is_blank_image`, shared with h3.py).

The prompt is written against SLOT numbers (`<Picture 3>` = image_3) and rewritten
to the ordinals core presents once the empties are gone, with h3.py's own
`rewrite_prompt_tags`. A tag naming an empty slot is dropped.

The conditioning is DELEGATED to core's PH3ImageConditioning rather than copied: its
tensor maths moves with ComfyUI and a private copy would drift.

No ComfyUI import at module scope, so `python ph3.py` runs the self-check in bare Python.
"""

try:
    from .h3 import MAX_RESOLUTION, is_blank_image, rewrite_prompt_tags
except ImportError:  # `python ph3.py`, no package
    from h3 import MAX_RESOLUTION, is_blank_image, rewrite_prompt_tags

MAX_IMAGES = 8
# Core's CANVAS_MULTIPLE: width/height snap to it.
CANVAS_MULTIPLE = 32


def collect_images(slots: dict):
    """Drop the empty `image_1..8` slots, renumber the survivors from 1.

    Returns `images` in core's Autogrow shape (`{"image_1": ..., "image_2": ...}`, core
    sorts by the trailing number) and `kept`, the ORIGINAL slot numbers, in order.
    """
    images, kept = {}, []
    for i in range(1, MAX_IMAGES + 1):
        image = slots.get(f"image_{i}")
        if is_blank_image(image):
            continue
        images[f"image_{len(images) + 1}"] = image
        kept.append(i)
    return images, kept


def picture_tag_map(kept) -> str:
    """Which tag each surviving slot became - the `ref_tags` output."""
    if not kept:
        return "(no pictures - this is a plain text-to-image run)"
    lines = [f"<Picture {n}> = image_{slot}" + (" (canvas)" if n == 1 else "")
             for n, slot in enumerate(kept, 1)]
    return "\n".join(lines)


class MpiPH3ImageConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_IMAGES + 1):
            role = "The CANVAS: the picture being edited; the output keeps its size and aspect." if i == 1 else "Reference picture."
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"{role} Cited in the prompt as <Picture {i}>. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."})
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Cite pictures by SLOT: <Picture 3> is whatever is wired into image_3. The node rewrites tags to the order core presents once empty slots are dropped (see ref_tags); a tag naming an empty slot is removed. With pictures, a short instruction gets \"Edit <Picture 1>. \" in front (core); one starting \"Edit <Picture\" or an upsampler brief is used as it is."}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": CANVAS_MULTIPLE, "tooltip": "0 uses the canvas width after scaling, or 2912 without pictures."}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": CANVAS_MULTIPLE, "tooltip": "0 uses the canvas height after scaling, or 2912 without pictures."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "ref_tags")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "P-H3 Image conditioning with all 8 picture slots exposed at once: image_1 is the canvas "
        "(the picture edited; the output keeps its size), image_2..8 are references. Empty slots "
        "are dropped and the rest renumbered, so ONE graph covers text-to-image and every edit "
        "combination instead of growing core's Autogrow input per run. Empty means nothing "
        "connected or an Mpi loader with block_if_empty OFF (a 1x1 image) - a real black image "
        "passes through. Write the prompt against SLOT numbers - <Picture 3> means image_3 - and "
        "the node rewrites them to what core presents, dropping any tag whose slot is empty. "
        "Conditioning is built by core's own PH3ImageConditioning (ComfyUI 0.44.0+), so it cannot drift."
    )
    FUNCTION = "doit"

    def doit(self, clip, vae, prompt, width=0, height=0, **kwargs):
        # Imported here, not at module scope: the pack must still load on a ComfyUI
        # older than 0.44.0, where PH3ImageConditioning does not exist.
        try:
            from comfy_extras.nodes_minimax_h3 import PH3ImageConditioning  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PH3ImageConditioning is missing - MpiPH3ImageConditioning needs ComfyUI 0.44.0 or newer."
            ) from exc

        images, kept = collect_images(kwargs)
        prompt = rewrite_prompt_tags(prompt, kept, [], [])
        output = PH3ImageConditioning.execute(
            clip=clip, prompt=prompt, width=width, height=height,
            vae=vae, images=images or None,
        )
        cond, latent = output.result
        return (cond, latent, picture_tag_map(kept))


UPSAMPLER_MAX_IMAGES = 10
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def upsampler_prompt(request: str, count: int, aspect_ratio: str = "1:1") -> str:
    """The prompt Pruna's upsampler graph builds, with one picture line per SURVIVOR.

    With pictures it is a whole chat turn (Qwen3.5's tokenizer passes a prompt that
    starts with `<|im_start|>` through as-is and binds each `<|image_pad|>` to the next
    image in order), so the count here and the batch can never disagree. Without
    pictures it is Pruna's text-to-image request, wrapped by the default template.
    """
    request = request.strip()
    if count == 0:
        return f"Aspect ratio: {aspect_ratio}\nRequest: {request}"
    pictures = "".join(f"<Picture {n}>: {VISION_BLOCK}" for n in range(1, count + 1))
    if not request.startswith("Edit <Picture"):
        request = f"Edit <Picture 1>. {request}"
    return f"<|im_start|>user\n{pictures}{request}<|im_end|>\n<|im_start|>assistant\n<think>\n"


class MpiPH3UpsamplerImages:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, UPSAMPLER_MAX_IMAGES + 1):
            role = "The CANVAS: scaled to 1 MP; every other picture is padded to its size." if i == 1 else "Reference picture."
            optional[f"image_{i}"] = ("IMAGE", {"tooltip": f"{role} Cited in the request as <Picture {i}>. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."})
        return {
            "required": {
                "request": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The user's edit request. Cite pictures by SLOT (<Picture 3> = image_3); tags are renumbered to the survivors and a tag naming an empty slot is removed. \"Edit <Picture 1>. \" is put in front unless the request already starts with \"Edit <Picture\"."}),
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Only used with no pictures (text-to-image): Pruna's upsampler is told the aspect ratio, e.g. 16:9."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("images", "prompt", "count")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Feeds P-H3 Image's prompt upsampler (TextGenerate on the p-h3-image_prompt_upsampler weight). "
        "Takes up to 10 picture slots, drops the empty ones (nothing connected, or an Mpi loader's 1x1 "
        "image), scales the canvas to 1 MP and pads the rest to its size in white, as Pruna's graph does, "
        "and emits the batch together with the matching chat prompt - one <Picture N> line per picture "
        "kept - so the two can never disagree. Wire images to TextGenerate's image input, prompt to its "
        "prompt, thinking ON. With no pictures, images is empty and prompt is Pruna's text-to-image request."
    )
    FUNCTION = "doit"

    def doit(self, request, aspect_ratio="1:1", **kwargs):
        import torch  # lazy: the self-check runs without torch
        from comfy_extras.nodes_images import ResizeAndPadImage  # type: ignore
        from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels  # type: ignore

        survivors, kept = [], []
        for i in range(1, UPSAMPLER_MAX_IMAGES + 1):
            image = kwargs.get(f"image_{i}")
            if is_blank_image(image):
                continue
            survivors.append(image[:1])
            kept.append(i)

        prompt = rewrite_prompt_tags(request, kept, [], [])
        if not survivors:
            return (None, upsampler_prompt(prompt, 0, aspect_ratio), 0)

        # Delegated to core so the sizing matches Pruna's graph exactly.
        canvas = ImageScaleToTotalPixels.execute(survivors[0], "lanczos", 1.0, 1).result[0]
        _, h, w, _ = canvas.shape
        batch = [canvas] + [
            ResizeAndPadImage.execute(img, w, h, "white", "lanczos").result[0].to(canvas)
            for img in survivors[1:]
        ]
        return (torch.cat(batch, dim=0), upsampler_prompt(prompt, len(batch), aspect_ratio), len(batch))


if __name__ == "__main__":
    class _Img:
        def __init__(self, h, w):
            self.shape = (1, h, w, 3)

    canvas, ref, blank = _Img(1024, 768), _Img(512, 512), _Img(1, 1)

    # All empty = text-to-image.
    images, kept = collect_images({"image_1": blank, "image_4": None})
    assert images == {} and kept == [], (images, kept)
    assert picture_tag_map(kept).startswith("(no pictures")

    # Canvas + refs with gaps: survivors renumbered from 1, canvas stays first.
    images, kept = collect_images({"image_1": canvas, "image_2": blank, "image_5": ref, "image_8": ref})
    assert kept == [1, 5, 8], kept
    assert list(images) == ["image_1", "image_2", "image_3"], images
    assert images["image_1"] is canvas

    p = rewrite_prompt_tags("Put the man from <Picture 5> into <Picture 1>, hat from <Picture 8>, not <Picture 2>.", kept, [], [])
    assert p == "Put the man from <Picture 2> into <Picture 1>, hat from <Picture 3>, not .", p
    assert picture_tag_map(kept) == "<Picture 1> = image_1 (canvas)\n<Picture 2> = image_5\n<Picture 3> = image_8"

    # Core's edit-prefix check matches "Edit <Picture"; a rewritten tag keeps that shape.
    assert rewrite_prompt_tags("Edit <Picture 1>. make it night", kept, [], []).startswith("Edit <Picture 1>.")

    # Empty canvas: the first surviving reference becomes the canvas.
    images, kept = collect_images({"image_3": ref})
    assert kept == [3] and list(images) == ["image_1"]

    # Upsampler prompt: one picture line per survivor, Pruna's exact wrapping.
    two = upsampler_prompt("Put the hat from <Picture 2> on the person in <Picture 1>.", 2)
    assert two == ("<|im_start|>user\n<Picture 1>: " + VISION_BLOCK + "<Picture 2>: " + VISION_BLOCK
                   + "Edit <Picture 1>. Put the hat from <Picture 2> on the person in <Picture 1>.<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n"), two
    assert upsampler_prompt("x", 3).count(VISION_BLOCK) == 3
    assert "Edit <Picture 1>. Edit" not in upsampler_prompt("Edit <Picture 1>. make it night", 1)
    assert upsampler_prompt("a ramen shop", 0, "16:9") == "Aspect ratio: 16:9\nRequest: a ramen shop"
    print("ph3.py self-check passed")
