"""MiniMax H3 frame timing and reference conditioning.

H3 only generates frame counts on a 17k+5 grid at 24 fps, so most durations are
not reachable: asking for 2 s gets you 2.33 s (56 frames). The core H3 nodes snap
`length` up internally and never tell you what they picked, which leaves the graph
(and the UI in front of it) guessing. `MpiH3Length` does the same arithmetic up
front and reports the truth.

`MpiH3References` re-shapes core's reference conditioning so a host app can drive
it — see the comment above that class.

Deliberately free of ComfyUI imports so the arithmetic stays testable in bare
Python; `python h3.py` runs the self-check at the bottom.
"""

import re

FPS = 24.0
_GRID = 17
_OFFSET = 5
# From the core node's own tooltip (comfy_extras/nodes_minimax_h3.py): default 124,
# "trained range is ~124-362". Outside it the model still runs, untested.
TRAINED_MIN = 124
TRAINED_MAX = 362


def snap_h3_frames(frames: int) -> int:
    """Nearest valid H3 frame count (n % 17 == 5), minimum 5.

    Core snaps UP, which maximises the error — 4 s asks for 96 frames and gets
    107 (4.46 s) when 90 (3.75 s) is closer. Nearest is never worse. The result
    is already valid, so the core node's own snap leaves it alone.
    """
    k = round((frames - _OFFSET) / _GRID)
    return max(_OFFSET, k * _GRID + _OFFSET)


class MpiH3Length:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.2,
                        "max": 150.0,
                        "step": 0.1,
                        "tooltip": "Wanted duration. H3 can only land on a 17k+5 frame grid at 24 fps, so the `seconds` output is what you will actually get.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("frames", "seconds", "in_trained_range")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = "Convert a wanted duration into a valid MiniMax H3 frame count. H3 only generates n % 17 == 5 frames at 24 fps, so exact whole seconds are mostly impossible - 8 s is the shortest one that lands. Wire `frames` into the H3 node's `length`, and show `seconds` in the UI instead of what was asked for. `in_trained_range` is false outside 124-362 frames (~5.2-15.1 s), where the model still runs but was not trained."
    FUNCTION = "doit"

    def doit(self, seconds: float):
        frames = snap_h3_frames(round(seconds * FPS))
        return (frames, frames / FPS, TRAINED_MIN <= frames <= TRAINED_MAX)


# ---------------------------------------------------------------------------
# Reference conditioning
# ---------------------------------------------------------------------------
#
# Core's MiniMaxH3ReferenceToVideo takes its references through AUTOGROW slots:
# the author adds one slot at a time and the API names them
# `ref_images.ref_image_0`, `ref_videos.ref_video_0`, and so on. A host app that
# compiles a saved workflow cannot grow a slot at inject time, so covering "which
# references did the user actually supply" would need one pre-authored branch per
# combination -- 2**18 of them. This node takes the whole surface as flat,
# always-present optional inputs and drops the empty ones itself.
#
# "Empty" is not guessed from content. MpiLoadImageFromPath / MpiLoadVideo /
# MpiLoadAudio with `block_if_empty` OFF emit a 1x1 image and a 1-sample waveform,
# neither of which a real reference can be. A genuinely black reference still
# reports its true size and a genuinely silent clip still has its full sample
# count, so both pass through untouched.
#
# The conditioning work itself is DELEGATED to the core node rather than copied.
# Its tensor maths moves with ComfyUI, and a private copy would drift silently.

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
# Mirrors nodes.MAX_RESOLUTION. Inlined so this module stays importable without
# ComfyUI; it is a widget clamp only, nothing downstream depends on the value.
MAX_RESOLUTION = 16384


def _is_blank_image(image) -> bool:
    """True for a missing input or the loaders' 1x1 empty-path sentinel."""
    if image is None:
        return True
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 3:
        return True
    return shape[1] == 1 and shape[2] == 1


def _is_blank_audio(audio) -> bool:
    """True for a missing input or the loaders' 1-sample silent sentinel."""
    if not isinstance(audio, dict):
        return True
    waveform = audio.get("waveform")
    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) < 1:
        return True
    return shape[-1] <= 1


def collect_refs(slots: dict) -> dict:
    """Drop the empty reference slots and renumber the survivors from 0.

    `slots` is the raw `ref_image_1..9` / `ref_video_1..3` /
    `ref_video_audio_1..3` / `ref_audio_1..3` mapping as ComfyUI passes it.

    Renumbering is not cosmetic: core reads these dicts in insertion order and
    pairs a video to its soundtrack by the TRAILING NUMBER of the key, so a gap
    left by a dropped slot would mis-pair every reference after it.

    Returns `packed` (the four dicts core wants) and `slots` (the ORIGINAL slot
    numbers that survived, for the tag map).
    """
    images, image_slots = {}, []
    for i in range(1, MAX_REF_IMAGES + 1):
        image = slots.get(f"ref_image_{i}")
        if _is_blank_image(image):
            continue
        images[f"ref_image_{len(images)}"] = image
        image_slots.append(i)

    videos, video_audios, video_slots = {}, {}, []
    for i in range(1, MAX_REF_VIDEOS + 1):
        frames = slots.get(f"ref_video_{i}")
        if _is_blank_image(frames):
            continue
        index = len(videos)
        videos[f"ref_video_{index}"] = frames
        soundtrack = slots.get(f"ref_video_audio_{i}")
        has_audio = not _is_blank_audio(soundtrack)
        if has_audio:
            video_audios[f"ref_video_audio_{index}"] = soundtrack
        video_slots.append((i, has_audio))

    audios, audio_slots = {}, []
    for i in range(1, MAX_REF_AUDIOS + 1):
        track = slots.get(f"ref_audio_{i}")
        if _is_blank_audio(track):
            continue
        audios[f"ref_audio_{len(audios)}"] = track
        audio_slots.append(i)

    return {
        "packed": (images, videos, video_audios, audios),
        "slots": (image_slots, video_slots, audio_slots),
    }


def ref_tag_map(image_slots, video_slots, audio_slots) -> str:
    """Which prompt tag each surviving slot became.

    Mirrors the presentation order core documents on MiniMaxH3ReferenceToVideo:
    images, then videos with each soundtrack's <Audio j> emitted right BEFORE its
    <Video k>, then standalone audio. Ordinals are 1-based per type and the audio
    sequence is SHARED -- a reference video's soundtrack consumes an <Audio j>, so
    a standalone clip after one is <Audio 2>, not <Audio 1>. Nothing can be written
    into the prompt without this map, which is why it is an output.

    `video_slots` is a list of (slot, has_audio) pairs.
    """
    pictures = videos = audios = 0
    lines = []
    for slot in image_slots:
        pictures += 1
        lines.append(f"<Picture {pictures}> = ref_image_{slot}")
    for slot, has_audio in video_slots:
        if has_audio:
            audios += 1
            lines.append(f"<Audio {audios}> = ref_video_audio_{slot}")
        videos += 1
        lines.append(f"<Video {videos}> = ref_video_{slot}")
    for slot in audio_slots:
        audios += 1
        lines.append(f"<Audio {audios}> = ref_audio_{slot}")
    return "\n".join(lines) if lines else "(no references - this is a plain t2va run)"


_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>")


def rewrite_prompt_tags(prompt, image_slots, video_slots, audio_slots) -> str:
    """Translate SLOT-numbered tags into the ordinals core actually presents.

    A host app numbers its reference wells: well 1, well 2, well 3. Core numbers
    the SURVIVORS, and its audio sequence is shared -- a reference video's
    soundtrack consumes an <Audio j> before its <Video k>, so a standalone clip
    sitting behind a sounded video is <Audio 2> even though it is in audio well 1.
    Worse, whether a video HAS a soundtrack is a property of the FILE, so the
    number moves depending on a fact nobody knows until decode time. An app cannot
    label its own wells correctly, and a user cannot either.

    So the wells stay the contract. The user writes the tag their chip shows and
    this rewrites it, here, where both numberings are known. A tag naming an empty
    well is DROPPED rather than passed through -- core would present no such label,
    and a dangling one is a reference the model is told to look for and cannot find.

    Video soundtracks are deliberately not addressable: they have no well of their
    own, and <Video k> already names the clip they came from.
    """
    mapping = {}
    for ordinal, slot in enumerate(image_slots, 1):
        mapping[("Picture", slot)] = ordinal
    for ordinal, (slot, _has_audio) in enumerate(video_slots, 1):
        mapping[("Video", slot)] = ordinal
    # Standalone audio starts after every soundtrack that actually materialised.
    consumed = sum(1 for _, has_audio in video_slots if has_audio)
    for ordinal, slot in enumerate(audio_slots, 1):
        mapping[("Audio", slot)] = consumed + ordinal

    def _sub(match):
        target = mapping.get((match.group(1), int(match.group(2))))
        return "" if target is None else f"<{match.group(1)} {target}>"

    return re.sub(r"[ \t]{2,}", " ", _TAG_RE.sub(_sub, prompt)).strip()


class MpiH3References:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_REF_IMAGES + 1):
            optional[f"ref_image_{i}"] = ("IMAGE", {"tooltip": "Reference image. Leave unconnected, or feed a loader with block_if_empty OFF, to skip it."})
        for i in range(1, MAX_REF_VIDEOS + 1):
            optional[f"ref_video_{i}"] = ("IMAGE", {"tooltip": "Reference video frames at 24 fps (2-15 s)."})
            optional[f"ref_video_audio_{i}"] = ("AUDIO", {"tooltip": f"Soundtrack for ref_video_{i}. Pairing is by this number, so wire it from the same loader."})
        for i in range(1, MAX_REF_AUDIOS + 1):
            optional[f"ref_audio_{i}"] = ("AUDIO", {"tooltip": "Standalone reference audio."})
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Address references by their SLOT number here: <Picture 1> is whatever is wired into ref_image_1, <Audio 2> into ref_audio_2. The node rewrites them to the ordinals core presents (see ref_tags) - so a tag keeps meaning the same input no matter which other slots are empty. A tag naming an empty slot is removed."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": MAX_RESOLUTION, "step": 32}),
                "length": ("INT", {"default": 124, "min": 5, "max": 3600, "step": 17, "tooltip": "Frame count at 24 fps (124 = ~5 s, trained range ~124-362). Feed it from MpiH3Length."}),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": "How each reference is scaled (aspect kept, never upscaled, no crop). 'match' fits it to the OUTPUT's pixel area. 'max' uses a 2048 short edge and is best for identity - use it for character sheets, where 'match' squashes the individual views past readability. 'max' is several times slower: reference tokens ride through every sampling step.",
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "ref_tags")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "MiniMax H3 reference conditioning with every slot exposed at once: 9 images, "
        "3 videos with paired soundtracks, 3 standalone audio. Empty slots are dropped "
        "and the rest renumbered, so ONE graph covers every combination instead of a "
        "branch per combination. An empty slot means either nothing connected or an Mpi "
        "loader with block_if_empty OFF (a 1x1 image / 1-sample waveform) - real black "
        "images and real silence are not mistaken for empty. The conditioning is built "
        "by core's own MiniMaxH3ReferenceToVideo. Write the prompt against the SLOT "
        "numbers - <Picture 1> means ref_image_1 - and the node rewrites them to the "
        "ordinals core presents, dropping any tag whose slot is empty. That is what keeps "
        "a tag pointing at the same input when a slot ahead of it is left empty, and it "
        "is the only way to be right about audio: a reference video's soundtrack consumes "
        "an <Audio j>, and whether the file HAS one is unknown until it is decoded. "
        "`ref_tags` reports the resulting map."
    )
    FUNCTION = "doit"

    def doit(self, clip, vae, audio_vae, prompt, width, height, length,
             ref_image_size="match", **kwargs):
        # Imported here, not at module scope: the pack must still load on a
        # ComfyUI older than 0.30.0, where H3 does not exist at all.
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MiniMax H3 nodes are missing - MpiH3References needs ComfyUI 0.30.0 or newer."
            ) from exc

        refs = collect_refs(kwargs)
        ref_images, ref_videos, ref_video_audios, ref_audios = refs["packed"]
        image_slots, video_slots, audio_slots = refs["slots"]
        # The prompt arrives addressing SLOTS (what the host app's chips show) and
        # leaves addressing core's ordinals. See rewrite_prompt_tags.
        prompt = rewrite_prompt_tags(prompt, image_slots, video_slots, audio_slots)

        output = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=prompt,
            width=width, height=height, length=length,
            ref_image_size=ref_image_size,
            ref_images=ref_images or None,
            ref_videos=ref_videos or None,
            ref_video_audios=ref_video_audios or None,
            ref_audios=ref_audios or None,
        )
        cond, latent = output.result
        return (cond, latent, ref_tag_map(image_slots, video_slots, audio_slots))


if __name__ == "__main__":
    # Self-check: the slot filtering and the tag map, with stand-ins for the
    # tensors so it runs under bare `python h3.py`. The renumbering is the part
    # worth pinning -- a gap in the video keys silently re-pairs soundtracks.
    class _Img:
        """Stands in for an IMAGE tensor: (batch, height, width, channels)."""
        def __init__(self, h, w):
            self.shape = (1, h, w, 3)

    class _Wave:
        """Stands in for an AUDIO waveform: (batch, channels, samples)."""
        def __init__(self, samples):
            self.shape = (1, 2, samples)

    def _audio(samples):
        return {"waveform": _Wave(samples), "sample_rate": 44100}

    BLANK_IMG, BLANK_AUD = _Img(1, 1), _audio(1)
    real_img, real_aud = _Img(512, 512), _audio(240000)

    assert _is_blank_image(BLANK_IMG) and _is_blank_image(None)
    assert not _is_blank_image(_Img(1, 512)), "a 1px-tall strip is still real media"
    assert _is_blank_audio(BLANK_AUD) and _is_blank_audio(None)
    assert not _is_blank_audio(real_aud), "a silent-but-real clip must pass through"

    # Slots 2 and 3 used, 1 empty: the survivors must renumber to 0,1 and video 3's
    # soundtrack must follow it to index 1, not stay at 3.
    out = collect_refs({
        "ref_image_1": BLANK_IMG, "ref_image_2": real_img, "ref_image_5": real_img,
        "ref_video_1": BLANK_IMG,
        "ref_video_2": real_img, "ref_video_audio_2": BLANK_AUD,
        "ref_video_3": real_img, "ref_video_audio_3": real_aud,
        "ref_audio_2": real_aud,
    })
    images, videos, video_audios, audios = out["packed"]
    assert list(images) == ["ref_image_0", "ref_image_1"], images
    assert list(videos) == ["ref_video_0", "ref_video_1"], videos
    assert list(video_audios) == ["ref_video_audio_1"], video_audios
    assert list(audios) == ["ref_audio_0"], audios

    image_slots, video_slots, audio_slots = out["slots"]
    assert image_slots == [2, 5] and video_slots == [(2, False), (3, True)]
    assert audio_slots == [2]

    # The soundtrack consumes <Audio 1>, so the standalone clip is <Audio 2>.
    tags = ref_tag_map(image_slots, video_slots, audio_slots)
    assert tags.splitlines() == [
        "<Picture 1> = ref_image_2",
        "<Picture 2> = ref_image_5",
        "<Video 1> = ref_video_2",
        "<Audio 1> = ref_video_audio_3",
        "<Video 2> = ref_video_3",
        "<Audio 2> = ref_audio_2",
    ], tags

    # Tag rewriting, on the SAME staging as above. The user writes what the chips
    # show -- picture wells 2 and 5, video wells 2 and 3, audio well 2 -- and each
    # lands on core's ordinal. Video well 3 carries a soundtrack, so it eats
    # <Audio 1> and the standalone clip in audio well 2 becomes <Audio 2>. Getting
    # this wrong points the model at the wrong reference, silently.
    got = rewrite_prompt_tags(
        "<Picture 2> meets <Picture 5> in <Video 2>, moving like <Video 3>, saying <Audio 2>",
        image_slots, video_slots, audio_slots)
    assert got == "<Picture 1> meets <Picture 2> in <Video 1>, moving like <Video 2>, saying <Audio 2>", got

    # A tag for an EMPTY well is dropped, not passed through: core presents no such
    # label, so leaving it tells the model to find a reference that is not there.
    assert rewrite_prompt_tags("a shot of <Picture 1> at dusk", image_slots, video_slots, audio_slots) \
        == "a shot of at dusk"

    # No references at all: every tag is empty, so every tag goes.
    assert rewrite_prompt_tags("<Audio 1> over <Video 1>", [], [], []) == "over"

    # A prompt with no tags is returned untouched (bar the strip).
    assert rewrite_prompt_tags("a woman walking", [2], [], []) == "a woman walking"

    # With NO sounded video the standalone clip keeps its own well number -- this is
    # the case that used to shift under the user depending on the video FILE.
    assert rewrite_prompt_tags("<Audio 1>", [], [(1, False)], [1]) == "<Audio 1>"
    assert rewrite_prompt_tags("<Audio 1>", [], [(1, True)], [1]) == "<Audio 2>"

    assert collect_refs({})["packed"] == ({}, {}, {}, {})
    assert snap_h3_frames(96) == 90 and snap_h3_frames(5) == 5
    print("h3.py self-check passed")
