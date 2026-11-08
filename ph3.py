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


def edit_canvas_size(width: int, height: int, canvas, max_megapixels: float):
    """The width/height an EDIT runs at, never above `max_megapixels` (0 = no cap).

    Pruna: edits are made at 4 MP; above that faces and hands go wrong. Core sizes an
    edit from width/height when set, else from the canvas (picture 1) shrunk to 2912^2,
    so a big photo would edit at 8.5 MP. Over the cap, the size is scaled down to it,
    aspect kept, each side floored to the canvas grid. Never scaled up. Within the cap,
    width/height come back unchanged (0 stays 0: core keeps the picture's own size).
    """
    if max_megapixels <= 0 or canvas is None:
        return width, height
    w, h = (width, height) if width and height else (canvas.shape[2], canvas.shape[1])
    cap = max_megapixels * 1_000_000
    if w * h <= cap:
        return width, height
    scale = (cap / (w * h)) ** 0.5
    return tuple(max(CANVAS_MULTIPLE, int(side * scale) // CANVAS_MULTIPLE * CANVAS_MULTIPLE) for side in (w, h))


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
        optional["edit_max_megapixels"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1, "tooltip": "With pictures (an edit), the output never goes above this many megapixels: width/height, or picture 1's own size when they are 0, is scaled down to it, aspect kept. Never scaled up. 0 = no cap. Text-to-image ignores it."})
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

    def doit(self, clip, vae, prompt, width=0, height=0, edit_max_megapixels=0.0, **kwargs):
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
        width, height = edit_canvas_size(width, height, images.get("image_1"), edit_max_megapixels)
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


# Pruna's 8-step distilled schedule for the P-H3 LoRA (their lora_edit workflow).
PRUNA_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)


def _sigma_at(pos: float) -> float:
    """Pruna's curve at a fractional step index (0 = 1.0, 8 = 0.0), linear between steps."""
    last = len(PRUNA_SIGMAS) - 1
    pos = min(max(pos, 0.0), float(last))
    i = min(int(pos), last - 1)
    f = pos - i
    return PRUNA_SIGMAS[i] + (PRUNA_SIGMAS[i + 1] - PRUNA_SIGMAS[i]) * f


def _pos_of(sigma: float) -> float:
    """Inverse of `_sigma_at`: where on Pruna's curve `sigma` sits."""
    for i in range(len(PRUNA_SIGMAS) - 1):
        hi, lo = PRUNA_SIGMAS[i], PRUNA_SIGMAS[i + 1]
        if lo <= sigma <= hi:
            return i + (hi - sigma) / (hi - lo)
    return 0.0 if sigma > PRUNA_SIGMAS[0] else float(len(PRUNA_SIGMAS) - 1)


def ph3_denoise_sigmas(denoise: float, steps: int, start_min: float, start_max: float) -> list:
    """Map denoise to a start sigma, then `steps` steps along Pruna's curve down to 0.

    Core's SplitSigmasDenoise keeps round(8*d) of Pruna's steps, and those crowd near
    1.0, so the knob does nothing until ~0.8 and repaints above ~0.85. Here denoise is
    linear in the START sigma (start_min at 0, start_max at 1), and the steps are spread
    evenly in step-index along the same curve, so every run keeps Pruna's shape.
    """
    start = start_min + (start_max - start_min) * min(max(denoise, 0.0), 1.0)
    p0, p1 = _pos_of(start), float(len(PRUNA_SIGMAS) - 1)
    steps = max(int(steps), 1)
    return [_sigma_at(p0 + (p1 - p0) * k / steps) for k in range(steps)] + [0.0]


class MpiPH3Sigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the input picture is repainted. Linear in the START sigma: 0 starts at start_min, 1 at start_max."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 32, "tooltip": "Steps from the start sigma down to 0, spread along Pruna's 8-step curve."}),
                "start_min": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Start sigma at denoise 0. Bench: 0.5 is a near copy, 0.6 barely changes the picture."}),
                "start_max": ("FLOAT", {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Start sigma at denoise 1. Bench: 0.98 is a full restyle that keeps only the rough layout; 1.0 = a text-to-image run."}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Image-to-image sigmas for P-H3 Image. For a restyle, leave image_1 of the P-H3 conditioning EMPTY: with the picture there the edit conditioning holds it and denoise barely moves. Pruna's LoRA is distilled to 8 steps that crowd near 1.0, so "
        "SplitSigmasDenoise gives a knob that does nothing until ~0.8 and repaints above ~0.85. This maps "
        "denoise linearly to a start sigma between start_min and start_max and walks `steps` steps down "
        "Pruna's own curve from there to 0. Wire into SamplerCustomAdvanced's sigmas."
    )
    FUNCTION = "doit"

    def doit(self, denoise, steps, start_min, start_max):
        import torch  # lazy: the self-check runs without torch
        return (torch.FloatTensor(ph3_denoise_sigmas(denoise, steps, start_min, start_max)),)


if __name__ == "__main__":
    # Sigmas: endpoints, monotone, on Pruna's curve.
    assert all(abs(_sigma_at(_pos_of(s)) - s) < 1e-9 for s in PRUNA_SIGMAS)
    for d in (0.0, 0.3, 0.5, 1.0):
        s = ph3_denoise_sigmas(d, 6, 0.88, 0.99)
        assert len(s) == 7 and s[-1] == 0.0 and all(a > b for a, b in zip(s, s[1:])), s
        assert abs(s[0] - (0.88 + 0.11 * d)) < 1e-9, s
    assert [round(x, 6) for x in ph3_denoise_sigmas(1.0, 8, 0.0, 1.0)] == list(PRUNA_SIGMAS)

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

    # Edit cap: never above it, never scaled up, t2i (no canvas) and 0 untouched.
    assert edit_canvas_size(0, 0, None, 4.2) == (0, 0)                      # t2i
    assert edit_canvas_size(2912, 2912, None, 4.2) == (2912, 2912)          # t2i at 8 MP
    assert edit_canvas_size(0, 0, _Img(4000, 3000), 0) == (0, 0)            # cap off
    assert edit_canvas_size(0, 0, _Img(1024, 768), 4.2) == (0, 0)           # small picture keeps its size
    assert edit_canvas_size(2048, 2048, _Img(1024, 768), 4.2) == (2048, 2048)
    w, h = edit_canvas_size(0, 0, _Img(3000, 4000), 4.2)                    # 12 MP photo, 4:3
    assert w * h <= 4.2e6 and w % 32 == 0 and h % 32 == 0 and abs(w / h - 4 / 3) < 0.02, (w, h)
    w, h = edit_canvas_size(3872, 2176, _Img(1024, 768), 4.2)               # custom size at 8 MP
    assert w * h <= 4.2e6 and abs(w / h - 3872 / 2176) < 0.02, (w, h)

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
