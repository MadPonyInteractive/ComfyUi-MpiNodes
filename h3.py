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


# ---------------------------------------------------------------------------
# Masked prefix -- continuing a clip without regenerating its tail
# ---------------------------------------------------------------------------
#
# To continue a clip, write its encoded tail into the FRONT of the target latent
# and protect it with a noise mask (0 = preserve, 1 = generate). The model then
# only ever generates the part after it. Core has every piece of this except the
# composition: `samplers.py` blends `x*mask + latent_image*(1-mask)` each step,
# `MiniMaxH3.scale_latent_inpaint` hands the preserved region back UNSCALED, and
# `extra_conds` splits a nested mask into the video and audio halves so
# `process_timestep` can zero the timestep on preserved tokens. So this node is
# arithmetic and slicing, not model code.
#
# The alternative -- attaching the tail as keyframe conditioning and letting the
# model regenerate it -- pays for those frames on every extend and needs a trim.
#
# EVERY rule below fails SILENTLY. A context length off the grid does not raise;
# it renders a clip that reads as a bad model.

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
AUDIO_LATENT_FPS = 40
# Valid context lengths are the frame counts that sit on H3's 17k+5 video grid
# AND divide by 3, so they also land on a whole audio latent step (40 Hz against
# 24 fps is 5/3 steps per frame). That is n = 51j + 39: 39, 90, 141, 192, ...
# 39 frames = 1.625 s = exactly 65 audio steps. Anything else leaves a fractional
# overhang and the audio seam drifts against the picture.
_CONTEXT_BASE = 39
_CONTEXT_STEP = 51


def is_valid_context(frames: int) -> bool:
    """True for the frame counts that land on BOTH clocks."""
    return frames >= _CONTEXT_BASE and (frames - _CONTEXT_BASE) % _CONTEXT_STEP == 0


def snap_context_frames(frames: int) -> int:
    """Largest valid context length <= `frames`, or 0 if none fits.

    DOWN, never nearest and never up. Snapping up would ask for context the clip
    does not have; the tail slice would then start before the clip does.
    """
    if frames < _CONTEXT_BASE:
        return 0
    return _CONTEXT_BASE + _CONTEXT_STEP * ((frames - _CONTEXT_BASE) // _CONTEXT_STEP)


def tail_span(total_steps: int, steps: int) -> int:
    """Pixel frames covered by the LAST `steps` latent steps of a clip.

    The video VAE packs frames per latent step on a period-5 cycle (1, 4, 4, 4,
    4), so a step's span depends on its ABSOLUTE index, not on its distance from
    the end. Counting from the end without that is trap 1 in the brief: ten
    reference frames encode to the same two latent steps as five, and the tail
    then covers frames [-10..-6] instead of [-5..-1]. The seam lands five frames
    early and the continuation carries on from an instant that never happened.
    """
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(max(0, total_steps - steps), total_steps))


def frame_spans(steps: int):
    """(start, end) pixel-frame range each latent step packs, step 0 first.

    The same period-5 packing `tail_span` counts from the other end, walked
    forwards. Anything mapping a PER-FRAME signal onto the latent grid has to
    walk this rather than divide by a constant: step 0 carries one frame and
    every other step carries four, so the two clocks never line up.
    """
    out, f = [], 0
    for k in range(steps):
        span = FRAME_PER_TOKEN[k % 5]
        out.append((f, f + span))
        f += span
    return out


def plan_context(total_steps: int, wanted_frames: int):
    """Pick the tail of a clip that is a valid context length.

    Returns `(steps, frames, audio_steps)`, or `(0, 0, 0)` when the clip is too
    short. Walks the step count DOWN and measures each candidate's true span, so
    it stays right for a source clip whose own length is off the grid -- which is
    any clip that came in through a video loader rather than a prior sample.
    """
    for steps in range(min(total_steps, max(1, wanted_frames)), 0, -1):
        frames = tail_span(total_steps, steps)
        if frames > wanted_frames:
            continue
        if is_valid_context(frames):
            return steps, frames, frames * AUDIO_LATENT_FPS // FPS_I
    return 0, 0, 0


FPS_I = 24


def plan_audio_window(start, end, frames, audio_steps):
    """A frame range -> the audio latent steps it covers, snapped to whole steps.

    Audio runs on a 40 Hz clock against 24 fps picture, so a frame boundary is a
    whole audio step only every 3 frames. An off-grid range raises nowhere: it
    lands the seam a fraction of a step out and the regenerated window drifts
    against the picture, the same silent failure as an off-grid frame count. So
    the range is snapped OUTWARDS -- start down, end up -- and the caller reports
    what it actually used.

    `end` of 0 means no audio window at all; -1 runs to the end of the clip, which
    needs no snap because it takes the audio latent whole (a clip length on the
    17k+5 grid is never a multiple of 3, so its last step is a fraction).

    Returns `(a0, a1, f0, f1)` -- audio steps and the snapped frames -- all zero
    when there is no window.
    """
    if end == 0:
        return 0, 0, 0, 0
    if start < 0:
        raise ValueError(f"`audio_start` cannot be negative; got {start}.")
    f0 = min(start - start % 3, frames)
    if end < 0:
        f1, a1 = frames, audio_steps
    else:
        f1 = min(end + (-end % 3), frames)
        a1 = min(f1 * AUDIO_LATENT_FPS // FPS_I, audio_steps)
    if f1 <= f0:
        raise ValueError(
            f"The audio window is empty: frames {start}..{end} snap to {f0}..{f1} "
            f"inside a {frames}-frame clip. audio_end has to come after "
            "audio_start, and audio_start has to fall inside the clip."
        )
    return f0 * AUDIO_LATENT_FPS // FPS_I, a1, f0, f1


def audio_at_vae_rate(audio, audio_vae):
    """The clip's waveform at the audio VAE's own rate, as [B, C, samples].

    The encode resamples before encoding, so the latent lives in this domain and
    anything spliced back into it has to be resampled the same way first.
    """
    waveform = audio["waveform"]
    rate = audio["sample_rate"]
    vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)
    if rate != vae_rate:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, rate, vae_rate)
    return waveform, vae_rate


def splice_audio(original, generated, s0, s1, crossfade, mode="replace"):
    """Put a regenerated sample range back into the original waveform.

    Both tensors are [B, C, samples] at the SAME rate; `s0`/`s1` are sample
    indices. The crossfade ramps INSIDE the window at each end, so nothing
    outside it is touched -- a hard splice clicks, and a click is the one
    artefact an audio inpaint cannot hide. Equal power (sqrt), not linear: the
    generated window is a different take, so the two sides do not correlate and a
    linear pair dips through the middle of the crossover.

    `mode` "replace" swaps the window out. "mix" sums the generated window UNDER
    the original instead, which is the only way to layer a sound onto an existing
    track: an audio mask selects a time range, not a source, so H3 regenerates
    every source inside it -- dialogue, room tone and footsteps together. Replace
    is right for a whole re-take and wrong for adding a servo under a line.
    """
    import torch

    out = original.clone()
    n = min(out.shape[-1], generated.shape[-1])
    s0, s1 = max(0, min(s0, n)), max(0, min(s1, n))
    if s1 <= s0:
        return out

    gen = generated[..., s0:s1].to(out)
    xf = max(0, min(crossfade, (s1 - s0) // 2))
    if xf:
        t = torch.linspace(0, 1, xf + 2, dtype=gen.dtype, device=gen.device)[1:-1]
        ramp = torch.ones(s1 - s0, dtype=gen.dtype, device=gen.device)
        ramp[:xf] = t
        ramp[-xf:] = t.flip(0)
        gen = gen * ramp.sqrt()
        if mode == "replace":
            gen = gen + out[..., s0:s1] * (1.0 - ramp).sqrt()
    if mode == "mix":
        out[..., s0:s1] = out[..., s0:s1] + gen
        return out.clamp_(-1.0, 1.0)
    out[..., s0:s1] = gen
    return out


class MpiH3MaskedPrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The TARGET AV latent, at the full output length - straight from MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo or EmptyMiniMaxH3LatentAV. Its canvas must match the context clip's."}),
                "context_latent": ("LATENT", {"tooltip": "The prior clip as an AV latent. Encode the whole context run in ONE VAE call - encoding frame by frame throws away the motion the temporal packing carries."}),
                "context_frames": (
                    "INT",
                    {
                        "default": 39,
                        "min": 39,
                        "max": 1000,
                        "step": 51,
                        "tooltip": "How much of the prior clip to keep, in frames at 24 fps. Snapped DOWN to 39 / 90 / 141 / 192 ... - the only lengths that land on the video grid AND on a whole audio step. 39 frames is 1.625 s.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "context_frames", "new_frames", "report")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Continue a MiniMax H3 clip by writing its encoded tail into the front of the "
        "target latent and masking that region out of sampling, so the model generates "
        "only what comes after it. Nothing is regenerated and nothing has to be trimmed, "
        "which is what separates this from the keyframe/guide route. `context_frames` is "
        "snapped DOWN to 39 / 90 / 141 ...: those are the only lengths that sit on H3's "
        "17k+5 video grid and also divide by 3, so audio's 40 Hz clock lands on a whole "
        "step too. Off-grid values do not raise, they drift the audio against the picture "
        "and start the continuation from a moment that never happened, so the snap is "
        "reported rather than assumed. The output latent carries a nested video+audio "
        "noise mask (0 preserve, 1 generate) that core's sampler already understands. "
        "Do NOT put a first-frame guide inside the preserved head - it fights the prefix "
        "that already owns those frames."
    )
    FUNCTION = "doit"

    def doit(self, latent, context_latent, context_frames):
        # Imported here, not at module scope, so the pack still loads on a
        # ComfyUI without H3 -- same reason as MpiH3References.
        import torch
        import comfy.nested_tensor

        target_v, target_a = _unpack_av(latent, "latent")
        ctx_v, ctx_a = _unpack_av(context_latent, "context_latent")

        if target_v.shape[0] != 1 or ctx_v.shape[0] != 1:
            raise ValueError("MiniMax H3 is batch size 1 on both streams.")
        if target_v.shape[3:] != ctx_v.shape[3:]:
            raise ValueError(
                "The context clip and the target must share a canvas: context is "
                f"{ctx_v.shape[4] * 16}x{ctx_v.shape[3] * 16}, target is "
                f"{target_v.shape[4] * 16}x{target_v.shape[3] * 16}."
            )

        wanted = snap_context_frames(context_frames)
        steps, frames, audio_steps = plan_context(ctx_v.shape[2], wanted)
        if steps == 0:
            aligned = ctx_v.shape[2] % 5 == 2
            raise ValueError(
                f"No valid context fits in this clip ({ctx_v.shape[2]} latent steps, "
                f"{tail_span(ctx_v.shape[2], ctx_v.shape[2])} frames). "
                + ("It is shorter than the 39-frame minimum." if aligned else
                   "Its frame count is off H3's 17k+5 grid, which shifts the VAE's "
                   "packing phase so no tail of it lands on a legal length. Trim the "
                   "source to 17k+5 frames (39, 56, 73, ...) BEFORE encoding it.")
            )
        if steps >= target_v.shape[2]:
            raise ValueError(
                f"A {frames}-frame context leaves no room in a target of "
                f"{tail_span(target_v.shape[2], target_v.shape[2])} frames."
            )
        if audio_steps > ctx_a.shape[-1] or audio_steps >= target_a.shape[-1]:
            raise ValueError(
                f"The audio streams cannot hold a {frames}-frame prefix ({audio_steps} steps): "
                f"context has {ctx_a.shape[-1]}, target has {target_a.shape[-1]}."
            )

        out_v = target_v.clone()
        out_a = target_a.clone()
        out_v[:, :, :steps] = ctx_v[:, :, ctx_v.shape[2] - steps:].to(out_v)
        out_a[..., :audio_steps] = ctx_a[..., ctx_a.shape[-1] - audio_steps:].to(out_a)

        # Built at the latent's own resolution on purpose. core's reshape_mask
        # INTERPOLATES a mask that does not match (trilinear for video, bilinear
        # for audio), which would smear the one boundary that has to stay hard.
        mask_v = torch.ones((1, 1) + tuple(out_v.shape[2:]), dtype=torch.float32)
        mask_a = torch.ones((1, 1) + tuple(out_a.shape[2:]), dtype=torch.float32)
        mask_v[:, :, :steps] = 0.0
        mask_a[..., :audio_steps] = 0.0

        new_frames = tail_span(out_v.shape[2], out_v.shape[2]) - frames
        report = (
            f"context {frames} frames ({steps} latent steps, {audio_steps} audio steps)"
            f" + {new_frames} new; asked for {context_frames}"
        )

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((out_v, out_a))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        return (out, frames, new_frames, report)


def latent_frames(steps: int) -> int:
    """Pixel frames a video latent of `steps` steps decodes back to."""
    return frame_spans(steps)[-1][1] if steps else 0


def place_mask(mask, frames, start=0, end=-1):
    """A mask covering PART of a clip -> a full-length [F, H, W] at frame rate.

    A segmenter run over a frame range returns masks for that range only, so its
    row 0 is frame `start`, not frame 0. Nothing downstream can notice the shift:
    the tensor is a valid mask either way, so an unplaced sub-range mask inpaints
    the wrong moment of the clip and returns a clean render of the wrong thing.
    Passing the whole clip's masks is just the `start = 0` case of this.

    `end` clips the tail, counted in whole-clip frames; -1 keeps all of it.
    """
    import torch

    if mask.ndim != 3:
        raise ValueError(
            f"`mask` must be a per-frame mask [F, H, W]; got {tuple(mask.shape)}."
        )
    if start < 0:
        raise ValueError(f"`mask_start` cannot be negative; got {start}.")
    if mask.shape[0] == 1 and start == 0 and frames > 1:
        # One mask for a whole clip means the same region on every frame - the only
        # other reading is "inpaint frame 0 and keep the other 123", which no caller
        # wants and which nothing downstream can notice. Anything placed at a
        # non-zero start IS a deliberate range, so it is left alone.
        mask = mask.expand(frames, -1, -1)
    if start + mask.shape[0] > frames:
        raise ValueError(
            f"A {mask.shape[0]}-frame mask placed at frame {start} runs past the "
            f"end of a {frames}-frame clip. Either the mask covers a different "
            f"range than mask_start says, or it came from a different clip."
        )

    out = torch.zeros((frames,) + tuple(mask.shape[1:]), dtype=torch.float32)
    out[start:start + mask.shape[0]] = mask.float()
    if end >= 0:
        out[end:] = 0.0
    return out


def pack_video_mask(mask, video_shape, start=0, end=-1):
    """Per-frame pixel mask -> the video latent's own (1, 1, T, H, W) grid.

    Built at latent resolution on purpose, the same reason MpiH3MaskedPrefix
    builds its own: core's reshape_mask INTERPOLATES a mask that does not match,
    and on the packed time axis that snaps each latent step to a SINGLE picked
    frame, so a mask covering only a few frames vanishes with no error at all.
    Here each step takes the UNION of the frames it really packs, and the spatial
    reduction is a max too, so a thin mask is not averaged below the strength it
    exists to carry.

    The mask may cover only part of the clip -- see `place_mask`.
    """
    import torch

    t, h, w = video_shape[2], video_shape[3], video_shape[4]
    m = place_mask(mask, latent_frames(t), start, end)
    m = torch.nn.functional.adaptive_max_pool2d(m.unsqueeze(1), (h, w))[:, 0]

    out = torch.zeros((1, 1, t, h, w), dtype=torch.float32)
    for k, (f0, f1) in enumerate(frame_spans(t)):
        out[0, 0, k] = m[f0:f1].amax(0)
    return out


def sampled_mask_frames(mask, video_shape, frames, size, start=0, end=-1, patch=2):
    """The mask the MODEL actually sampled, brought back to pixel resolution.

    `pack_video_mask` quantises to the latent grid and unions the frames each
    latent step packs, and the DiT then pools again with amax over its own patch
    (`_pool_masks_to_token_grid` in comfy/model_base.py). So the region that was
    PAINTED is blockier and fatter than the per-frame mask -- on a 1536x640 clip
    the model sees 48x20 cells, each smeared over the ~4 frames its step carries.
    Compositing that repaint through the thin per-frame mask keeps a slice of it
    and puts the seam INSIDE the region the model worked to, which is exactly
    where a seam shows. This walks the same reductions and expands them back, so
    the kept region matches the painted one.
    """
    import torch

    t = video_shape[2]
    packed = pack_video_mask(mask, video_shape, start, end)[0, 0]          # (t, h, w)

    if patch > 1:  # the DiT's own amax over its 2x2 patch, then back to latent cells
        lh, lw = packed.shape[-2:]
        p = torch.nn.functional.pad(
            packed.unsqueeze(1), (0, -lw % patch, 0, -lh % patch), mode="replicate")
        p = torch.nn.functional.max_pool2d(p, patch, stride=patch)
        p = p.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
        packed = p[:, 0, :lh, :lw]

    spans = frame_spans(t)
    assert spans[-1][1] == frames, (spans[-1][1], frames)
    out = torch.zeros((frames,) + tuple(packed.shape[-2:]), dtype=torch.float32)
    for k, (f0, f1) in enumerate(spans):
        out[f0:f1] = packed[k]

    # nearest, not bilinear: the blockiness IS the mask the model was given
    return torch.nn.functional.interpolate(
        out.unsqueeze(1), size=size, mode="nearest")[:, 0]


class MpiH3EncodeAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": "The clip's frames at 24 fps, as ONE batch - encoding them in separate calls throws away the motion the temporal packing carries."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE."}),
                "audio": ("AUDIO", {"tooltip": "The clip's soundtrack. Resampled to the audio VAE's own rate if it does not already match."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Per-frame inpainting mask at image resolution, e.g. from SAM3 (1 = regenerate, 0 = keep). It may cover the WHOLE clip or just a range of it - say where its first frame sits with mask_start. Left unconnected this is a plain encode with no mask, exactly as before."}),
                "mask_start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "The clip frame the mask's FIRST frame lands on. Leave at 0 when the mask covers the whole clip; set it to the range start when the mask came from a segmenter run over part of the clip. Frames outside the mask are kept."}),
                "mask_end": ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip": "Clip frame to stop masking at, exclusive. -1 uses all of the supplied mask, so the default range is whatever the mask covers."}),
                "audio_start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "First clip frame of the AUDIO window to regenerate, at 24 fps. Snapped DOWN to a multiple of 3 - audio's 40 Hz clock only lands on a whole step every 3 frames, and an off-grid edge drifts the sound against the picture with no error. Only used when audio_end is not 0."}),
                "audio_end": ("INT", {"default": 0, "min": -1, "max": 100000, "tooltip": "Clip frame to stop regenerating audio at, exclusive; snapped UP to a multiple of 3. 0 - the default - keeps the whole soundtrack, which is the plain inpaint. -1 runs to the end of the clip. Unlike the picture mask this selects a TIME range, not a source: everything sounding in the window is regenerated together, so it gives 'redo this window to match the video', never 'add a sound under the dialogue' - for that, decode in mix mode."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Encode a clip and its soundtrack into ONE MiniMax H3 AV latent, which is "
        "what MpiH3MaskedPrefix takes as its context. Core encodes the two streams "
        "separately (VAEEncode + VAEEncodeAudio) and exposes no way to join them, so "
        "without this the only route to a joint latent is a third-party fork. "
        "With a `mask` it is also the inpainting encode: the mask is packed onto the "
        "video latent's own grid, so a host app reaches H3 inpainting from IMAGE + "
        "AUDIO + MASK, with no VIDEO type and no file loader in the graph. The mask "
        "may cover only PART of the clip - which is what a segmenter run over a "
        "frame range returns - as long as mask_start says where it begins. "
        "`audio_start`/`audio_end` open a window in the SOUNDTRACK the same way, "
        "which is context-aware foley: the model hears the rest of the track, so "
        "what it generates matches the room and the mic and is in sync by "
        "construction. The two masks are independent - an audio window with no "
        "picture mask regenerates sound over untouched video. Pair it with "
        "MpiH3DecodeAV."
    )
    FUNCTION = "doit"

    def doit(self, vae, images, audio_vae, audio, mask=None, mask_start=0, mask_end=-1,
             audio_start=0, audio_end=0):
        # Imported here, not at module scope, so the pack still loads on a
        # ComfyUI without H3 -- same reason as MpiH3References.
        import torch
        import comfy.nested_tensor

        video_z = vae.encode(images[..., :3])

        waveform, _vae_rate = audio_at_vae_rate(audio, audio_vae)
        # Batch 1 only: H3 is batch size 1 on both streams, and a second item
        # here would pair the wrong soundtrack to the picture rather than fail.
        audio_z = audio_vae.encode(waveform[:1].movedim(1, -1))

        out = {"samples": comfy.nested_tensor.NestedTensor((video_z, audio_z))}
        a0, a1, f0, f1 = plan_audio_window(
            audio_start, audio_end, images.shape[0], audio_z.shape[-1])
        if mask is not None or a1 > a0:
            # Only the inpainting path needs this: the mask and the composite in
            # MpiH3DecodeAV are indexed in PIXEL frames, so the clip has to survive
            # the round trip frame-for-frame. Off the grid the VAE packs what it can
            # and drops the remainder without complaining -- 6 frames come back as 5
            # - which then surfaces somewhere else entirely, or not at all. The
            # context path (MpiH3MaskedPrefix) deliberately accepts off-grid clips,
            # so this cannot be a check on every encode.
            if images.shape[0] % 17 != 5:
                raise ValueError(
                    f"An inpaint needs a frame count on H3's 17k+5 grid, and "
                    f"{images.shape[0]} is not one - the VAE would pack it to "
                    f"{latent_frames(video_z.shape[2])} frames and drop the rest. "
                    f"Use {snap_h3_frames(images.shape[0])} frames "
                    f"(5, 22, 39, 56, 73, ...); MpiH3Length picks one for you."
                )
            # Both halves default to zeros -- keep everything -- and each is sized
            # off its own encoded latent, so nothing here needs a frame count or a
            # frame rate to get the shapes right. Either half may stay all-keep:
            # a picture mask with no audio window is the plain inpaint, and an
            # audio window with no picture mask is foley over untouched video.
            mask_v = (
                pack_video_mask(mask, video_z.shape, mask_start, mask_end)
                if mask is not None
                else torch.zeros((1, 1) + tuple(video_z.shape[2:]), dtype=torch.float32)
            )
            mask_a = torch.zeros((1, 1) + tuple(audio_z.shape[2:]), dtype=torch.float32)
            mask_a[..., a0:a1] = 1.0
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))

        parts = []
        parts.append(
            f"picture: mask over frames {mask_start}.."
            f"{mask_end if mask_end >= 0 else images.shape[0]}"
            if mask is not None else "picture: no mask"
        )
        parts.append(
            f"audio: regenerating frames {f0}..{f1} "
            f"(steps {a0}..{a1} of {audio_z.shape[-1]})"
            if a1 > a0 else "audio: keeping all"
        )
        return (out, "; ".join(parts))


class MpiH3DecodeAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The sampled AV latent."}),
                "vae": ("VAE", {"tooltip": "The H3 video VAE."}),
                "images": ("IMAGE", {"tooltip": "The ORIGINAL frames the encode was given. Everything outside the mask is taken from here, not from the decode."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "The SAME mask the encode was given (1 = regenerate, 0 = keep). Unconnected returns the decode whole, with no compositing."}),
                "mask_start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Must match the encode's mask_start."}),
                "mask_end": ("INT", {"default": -1, "min": -1, "max": 100000, "tooltip": "Must match the encode's mask_end."}),
                "feather": ("INT", {"default": 11, "min": 1, "max": 101, "step": 2, "tooltip": "Softens the mask edge, in pixels, so the seam does not show (1 = hard cut). Odd only: the kernel is centred on the pixel, and min 0 with step 2 put every reachable value on the EVEN grid, so the arrows could never return to the default 11. Dilate then Gaussian, matching the LanPaint blend_overlap: the ramp sits OUTSIDE the mask you drew, so the inpainted content keeps full strength at its own edge."}),
                "mask_mode": (["per-frame", "as sampled"], {"default": "per-frame", "tooltip": "Which mask to composite through. `per-frame` keeps the exact mask you drew - tightest result, but the model painted a blockier, temporally wider region than that, so the seam lands inside the repaint. `as sampled` rebuilds the mask the model was actually given (latent grid, union of the frames each step packs, the DiT's own patch pooling) so the kept region matches the painted one; the boundary is blockier but it falls where the model stopped working, and feather can come right down."}),
                "audio_vae": ("VAE", {"tooltip": "The H3 audio VAE. Wire it only to decode a regenerated audio window - without it the soundtrack is passed through untouched, which is what a picture-only inpaint wants."}),
                "audio": ("AUDIO", {"tooltip": "The ORIGINAL soundtrack the encode was given. Everything outside the audio window is taken from here, not from a VAE round trip of it."}),
                "audio_start": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Must match the encode's audio_start."}),
                "audio_end": ("INT", {"default": 0, "min": -1, "max": 100000, "tooltip": "Must match the encode's audio_end. 0 passes the original soundtrack straight through."}),
                "audio_mode": (["replace", "mix"], {"default": "replace", "tooltip": "`replace` swaps the window for what the model generated - right for a whole re-take of that moment. `mix` sums it UNDER the original instead, which is the only way to layer a new sound onto a track that already has dialogue: an audio mask selects a time range, not a source, so the model regenerates everything sounding in the window and replacing it throws the original performance away."}),
                "audio_crossfade": ("INT", {"default": 50, "min": 0, "max": 2000, "tooltip": "Length of the crossfade at each end of the window, in milliseconds. A hard splice clicks, and a click is the one artefact an audio edit cannot hide. The ramp sits INSIDE the window, so the kept soundtrack is never touched. This is the audio half of LanPaint's pair - the counterpart to feather."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Decode the video half of a MiniMax H3 AV latent and composite it back into "
        "the original frames through the inpainting mask, which is what finishes the "
        "job MpiH3EncodeAV starts. Everything outside the mask is the ORIGINAL pixel, "
        "not a VAE round trip of it, so an inpaint does not quietly soften the whole "
        "frame, and the same holds for the soundtrack: with no audio_vae the audio "
        "input is passed through untouched, so a picture-only inpaint needs no audio "
        "wiring at all. Give it the audio_vae and the encode's audio window and it "
        "decodes the audio half too and splices the regenerated stretch back in over "
        "a crossfade. `feather` and `audio_crossfade` are the two halves of "
        "LanPaint's pair - dilate-then-Gaussian like its blend_overlap, and an "
        "equal-power fade like its audio_crossfade. `mask_mode` chooses whether to "
        "composite through the mask you drew or the coarser one the model was "
        "actually given; `audio_mode` whether the window replaces the original or "
        "layers on top of it."
    )
    FUNCTION = "doit"

    def doit(self, samples, vae, images, mask=None, mask_start=0, mask_end=-1, feather=11,
             mask_mode="per-frame", audio_vae=None, audio=None, audio_start=0, audio_end=0,
             audio_mode="replace", audio_crossfade=50):
        import torch

        z_video, z_audio = _unpack_av(samples, "samples")
        out = vae.decode(z_video)
        if out.ndim == 5:  # [1, F, H, W, C]: fold the batch into the frames
            out = out.reshape(-1, *out.shape[-3:])

        # A VAE decode can round the canvas; the composite needs an exact match.
        h, w = images.shape[1], images.shape[2]
        if out.shape[1:3] != (h, w):
            out = torch.nn.functional.interpolate(
                out.movedim(-1, 1), size=(h, w), mode="bilinear", align_corners=False
            ).movedim(1, -1)

        if out.shape[0] != images.shape[0]:
            raise ValueError(
                f"The latent decodes to {out.shape[0]} frames but {images.shape[0]} "
                "original frames were given. H3 only generates 17k+5 frames, so a "
                "clip trimmed off that grid cannot round-trip - snap it with "
                "MpiH3Length before encoding."
            )

        out_audio = self.compose_audio(
            z_audio, audio_vae, audio, images.shape[0], audio_start, audio_end,
            audio_mode, audio_crossfade)

        if mask is None:
            return (out, out_audio)

        if mask_mode == "as sampled":
            m = sampled_mask_frames(
                mask, z_video.shape, images.shape[0], (h, w), mask_start, mask_end)
        else:
            m = place_mask(mask, images.shape[0], mask_start, mask_end)
        m = m.unsqueeze(1)
        if feather > 1:
            k = feather | 1  # an even kernel shifts the mask half a pixel sideways
            # Dilate BEFORE smoothing, the way LanPaint's MaskBlend does. That puts
            # the whole ramp OUTSIDE the mask that was drawn, so the inpainted
            # content holds full strength right up to its own edge. Smoothing alone
            # ramps inward as well and fades the new content back into the old,
            # which reads as the inpaint not having taken at the boundary.
            # Max is separable too, and the k*k form was the whole cost of a wide
            # feather: at k=61 that is 3721 comparisons per pixel, over every frame,
            # which reads as a hung node rather than a slow one. Two 1-D passes are
            # 2k = 122 and give a bit-identical result.
            m = torch.nn.functional.max_pool2d(m, (k, 1), stride=1, padding=(k // 2, 0))
            m = torch.nn.functional.max_pool2d(m, (1, k), stride=1, padding=(0, k // 2))
            x = torch.arange(k, dtype=torch.float32) - k // 2
            g = torch.exp(-(x ** 2) / (2 * ((k - 1) / 4) ** 2))
            g = g / g.sum()
            # Replicate, not zero, padding: a mask touching the frame edge - a
            # subject walking out of shot - would otherwise be faded away there.
            # Separable here as well, so the cost is 2k rather than k*k per pixel.
            m = torch.nn.functional.pad(m, (k // 2,) * 4, mode="replicate")
            m = torch.nn.functional.conv2d(m, g.view(1, 1, k, 1))
            m = torch.nn.functional.conv2d(m, g.view(1, 1, 1, k))
        m = m.movedim(1, -1).to(out)  # [F, H, W, 1], broadcast over channels

        return (images.to(out) * (1.0 - m) + out * m, out_audio)

    @staticmethod
    def compose_audio(z_audio, audio_vae, audio, frames, start, end, mode, crossfade_ms):
        """The soundtrack this node hands on: passed through, or spliced.

        Deliberately NOT core's `vae_decode_audio`: that divides the decode by its
        own standard deviation to keep a bare decode from clipping, which puts it
        in a different loudness domain from the original waveform it is about to
        be spliced into. Here the kept audio is the reference, so the decode is
        taken raw and the levels line up on their own.
        """
        import torch

        if audio is None:
            # Something has to come out of an AUDIO socket; this is the same
            # 1-sample sentinel the loaders use for "nothing here".
            return {"waveform": torch.zeros((1, 2, 1)), "sample_rate": 44100}
        if audio_vae is None or end == 0:
            return audio

        wave, rate = audio_at_vae_rate(audio, audio_vae)
        wave = wave[:1]
        gen = audio_vae.decode(z_audio).movedim(-1, 1)
        if wave.shape[1] == 1 and gen.shape[1] > 1:
            wave = wave.expand(-1, gen.shape[1], -1)

        steps = z_audio.shape[-1]
        per_step = gen.shape[-1] // max(1, steps)
        a0, a1, _f0, _f1 = plan_audio_window(start, end, frames, steps)
        if wave.shape[-1] < a1 * per_step:
            raise ValueError(
                f"The soundtrack is {wave.shape[-1]} samples but the audio window "
                f"ends at {a1 * per_step}. It is shorter than the clip, so the "
                "window would be silently truncated - give this node the same "
                "audio the encode was given."
            )
        spliced = splice_audio(
            wave.to(gen), gen, a0 * per_step, a1 * per_step,
            int(rate * crossfade_ms / 1000), mode)
        return {"waveform": spliced, "sample_rate": rate}


def _unpack_av(latent, name: str):
    """Split an H3 AV latent into its video and audio tensors."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        raise ValueError(f"`{name}` is not a latent.")
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            f"`{name}` carries video only. MiniMax H3 needs a joint video+audio latent - "
            "give the encode node an audio_vae and the clip's soundtrack."
        )
    return samples.tensors[0], samples.tensors[1]


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

    # The packing walk pack_video_mask rides on. A step's span depends on its
    # ABSOLUTE index, so walking forwards must agree with tail_span counting back;
    # get this wrong and a mask lands on frames the user never painted.
    assert frame_spans(1) == [(0, 1)]
    assert frame_spans(6) == [(0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 18)]
    for _t in (1, 5, 6, 12, 27):
        assert frame_spans(_t)[-1][1] == tail_span(_t, _t), _t
    # 39 frames -- the shortest valid context -- is 12 latent steps.
    assert frame_spans(12)[-1][1] == 39
    assert latent_frames(12) == 39 and latent_frames(0) == 0
    # What the inpaint guard rests on. Two different constraints meet here and
    # only one of them is the grid: the VAE round-trips more lengths than the
    # MODEL accepts (1, 9, 13, 17, 18 ... all pack exactly), so `n % 17 == 5` is
    # the stricter of the two and has to be a SUBSET of what packs exactly --
    # otherwise the guard would pass a clip the decode then cannot line up.
    _exact = {latent_frames(t) for t in range(1, 200)}
    for _n in range(5, 400, 17):
        assert _n in _exact, _n
    # ...and the counts the guard rejects really are the ones that lose frames.
    assert 6 not in _exact and latent_frames(2) == 5, 'a 6-frame clip packs to 5'
    assert snap_h3_frames(6) == 5 and snap_h3_frames(56) == 56

    # --- masked prefix: the arithmetic that fails silently -----------------
    # The only context lengths on BOTH clocks. 39 / 90 / 141 are the ones the
    # brief names; the rest of the family follows the same +51 step.
    assert [n for n in range(1, 200) if is_valid_context(n)] == [39, 90, 141, 192]
    # Every one of them is on the 17k+5 video grid AND a whole audio step.
    for n in (39, 90, 141, 192):
        assert n % 17 == 5 and n % 3 == 0
        assert n * AUDIO_LATENT_FPS % FPS_I == 0
    assert 39 * AUDIO_LATENT_FPS // FPS_I == 65      # the brief's worked example

    # Snapping is DOWN. 89 frames of context must become 39, never 90 -- 90 is
    # context the clip may not have.
    assert snap_context_frames(89) == 39
    assert snap_context_frames(90) == 90
    assert snap_context_frames(141) == 141
    assert snap_context_frames(38) == 0

    # A step's span depends on its absolute index, not its distance from the end.
    # A clip on the grid has 5k+2 steps, and its trailing 5 steps are always 17
    # frames -- but the FIRST step of any clip is only 1 frame.
    assert tail_span(2, 2) == 5 and tail_span(7, 7) == 22 and tail_span(12, 12) == 39
    assert tail_span(37, 37) == 124                  # the core node's default
    assert tail_span(37, 12) == 39 and tail_span(37, 27) == 90
    assert tail_span(37, 5) == 17

    # A 124-frame source gives a 39-frame context out of its last 12 steps.
    assert plan_context(37, 39) == (12, 39, 65)
    # Asking for more than the clip holds yields the largest that fits, not an error.
    assert plan_context(12, 141) == (12, 39, 65)
    # A clip too short for any valid context reports so instead of guessing.
    assert plan_context(7, 39) == (0, 0, 0)
    # A 73-frame source (17*4+5, so 22 latent steps) is what a real prior clip
    # looks like, and 39 frames of it is a legal context.
    assert tail_span(22, 22) == 73 and plan_context(22, 39) == (12, 39, 65)

    # OFF-GRID SOURCE: 30 steps is not 5k+2, so the packing phase is shifted and
    # NO tail of it lands on a valid context length -- the reachable spans run
    # 4, 8, 12, 16, 17, 21, ... and never hit 39 or 90. This is a real constraint,
    # not a rounding nuisance: a clip must be trimmed to 17k+5 frames BEFORE it is
    # encoded, or it cannot be continued at all. Reporting nothing is the correct
    # answer here; the node raises rather than picking a near-miss.
    assert plan_context(30, 90) == (0, 0, 0)
    assert 39 not in {tail_span(30, s) for s in range(1, 31)}

    import torch

    # the separable dilate must equal the k*k one it replaced, at every k
    for _k in (3, 11, 61):
        _m = (torch.rand(2, 1, 40, 50) > 0.85).float()
        _sq = torch.nn.functional.max_pool2d(_m, _k, stride=1, padding=_k // 2)
        _sep = torch.nn.functional.max_pool2d(_m, (_k, 1), stride=1, padding=(_k // 2, 0))
        _sep = torch.nn.functional.max_pool2d(_sep, (1, _k), stride=1, padding=(0, _k // 2))
        assert torch.equal(_sq, _sep), _k

    # sampled_mask_frames: the painted region, not the drawn one
    _F = latent_frames(7)                       # 7 latent steps
    _msk = torch.zeros((_F, 64, 64))
    _msk[0, 20:24, 20:24] = 1.0                 # a tiny mark on frame 0 only
    _shape = (1, 24, 7, 4, 4)                   # latent 4x4 -> each cell is 16px
    _as = sampled_mask_frames(_msk, _shape, _F, (64, 64))
    _pf = place_mask(_msk, _F)
    assert _as.shape == _pf.shape == (_F, 64, 64)
    assert (_as >= _pf).all()                   # never keeps less than the drawn mask
    assert _as[0].sum() > _pf[0].sum()          # quantised out to whole cells
    assert set(_as[0].unique().tolist()) <= {0.0, 1.0}
    # frame 0 is its own latent step, so the mark must NOT bleed into step 1's frames
    assert _as[1].sum() == 0, _as[1].sum()
    # and a mark inside a 4-frame step covers every frame of that step
    _msk2 = torch.zeros((_F, 64, 64)); _msk2[2, 20:24, 20:24] = 1.0
    _as2 = sampled_mask_frames(_msk2, _shape, _F, (64, 64))
    _f0, _f1 = frame_spans(7)[1]
    assert _as2[_f0:_f1].sum() == _as2[_f0].sum() * (_f1 - _f0)
    print("sampled_mask_frames ok")

    # place_mask: one mask covers the whole clip rather than frame 0 alone
    import torch
    _one = torch.ones((1, 8, 8))
    assert place_mask(_one, 22).shape == (22, 8, 8)
    assert place_mask(_one, 22)[21].all()
    # but a deliberate placement stays a placement
    assert not place_mask(_one, 22, start=5)[6].any()

    # --- audio window: the second clock, and the splice ---------------------
    # 0 means off, and off must stay off - this is the default on both nodes, so
    # a bug here would put an audio mask on every plain inpaint.
    assert plan_audio_window(0, 0, 73, 121) == (0, 0, 0, 0)
    # 24 frames = 1 s = 40 audio steps, and both edges already sit on the grid.
    assert plan_audio_window(24, 48, 73, 121) == (40, 80, 24, 48)
    # Off-grid edges snap OUTWARDS, never inwards: 25..47 covers 24..48. Snapping
    # in would leave a fraction of a step regenerating on one side of the seam.
    assert plan_audio_window(25, 47, 73, 121) == (40, 80, 24, 48)
    # -1 takes the latent whole. A 73-frame clip is 121.67 steps, so computing the
    # end from the frame count instead would drop the last fraction of a step.
    assert plan_audio_window(0, -1, 73, 121) == (0, 121, 0, 73)
    # A window past the end is clamped to the clip, not to a step that isn't there.
    assert plan_audio_window(60, 200, 73, 121) == (100, 121, 60, 73)
    # A window smaller than a step is widened to one, not rejected: outward is
    # outward, and 0..2 is a request for the first three frames of sound.
    assert plan_audio_window(0, 2, 73, 121) == (0, 5, 0, 3)
    assert plan_audio_window(10, 11, 73, 121) == (15, 20, 9, 12)
    # A REVERSED range is the one that cannot be snapped into anything.
    for _bad in ((50, 10), (200, 210)):
        try:
            plan_audio_window(_bad[0], _bad[1], 73, 121)
        except ValueError:
            pass
        else:
            raise AssertionError(f"an empty window must raise: {_bad}")

    # The splice: outside the window the original survives EXACTLY. That is the
    # audio half of "everything outside the mask is the original pixel".
    _orig = torch.full((1, 2, 1000), 0.5)
    _gen = torch.full((1, 2, 1000), -0.25)
    _sp = splice_audio(_orig, _gen, 400, 600, 0)
    assert torch.equal(_sp[..., :400], _orig[..., :400])
    assert torch.equal(_sp[..., 600:], _orig[..., 600:])
    assert torch.equal(_sp[..., 400:600], _gen[..., 400:600])
    assert torch.equal(_orig, torch.full((1, 2, 1000), 0.5)), "the input was mutated"

    # With a crossfade the ramp lives INSIDE the window - the kept audio is still
    # untouched - and the middle is still fully the generated take.
    _sp = splice_audio(_orig, _gen, 400, 600, 50)
    assert torch.equal(_sp[..., :400], _orig[..., :400])
    assert torch.equal(_sp[..., 600:], _orig[..., 600:])
    assert torch.allclose(_sp[..., 500], _gen[..., 500])
    assert not torch.allclose(_sp[..., 410], _gen[..., 410]), "no fade at the edge"
    # Equal power, measured on the GAIN PAIR rather than on a mixed signal: run
    # the splice once with only the original and once with only the generated
    # take, and the two gains must square-sum to 1 across the whole crossover.
    # That is the property sqrt buys - a linear pair sums to 1 unsquared and dips
    # ~3 dB through the middle when the two sides do not correlate.
    _zero = torch.zeros((1, 2, 1000))
    _keep = splice_audio(torch.ones((1, 2, 1000)), _zero, 400, 600, 50)
    _new = splice_audio(_zero, torch.ones((1, 2, 1000)), 400, 600, 50)
    _power = _keep[..., 400:600] ** 2 + _new[..., 400:600] ** 2
    assert torch.allclose(_power, torch.ones_like(_power), atol=1e-6), _power.min()

    # mix layers instead of replacing, and clamps rather than wrapping round.
    _sp = splice_audio(_orig, _gen, 400, 600, 0, mode="mix")
    assert torch.allclose(_sp[..., 500], _orig[..., 500] + _gen[..., 500])
    assert splice_audio(_orig, torch.ones_like(_gen), 0, 1000, 0, mode="mix").max() <= 1.0
    # A crossfade longer than the window is clamped, not an error.
    assert splice_audio(_orig, _gen, 400, 410, 10_000).shape == _orig.shape

    # MpiH3DecodeAV.compose_audio end to end, on a stand-in VAE -- the widgets
    # and the sample arithmetic are the half that a graph cannot check for you.
    class _AVae:
        audio_sample_rate = 32000

        def decode(self, z):                    # core hands back [B, samples, C]
            return torch.full((1, z.shape[-1] * 800, 2), -0.75)

    _z = torch.zeros((1, 32, 2, 121))           # 121 steps = a 73-frame clip
    _track = {"waveform": torch.full((1, 2, 121 * 800), 0.5), "sample_rate": 32000}
    _compose = MpiH3DecodeAV.compose_audio

    # No soundtrack at all still fills the socket, with the loaders' sentinel.
    assert _compose(_z, None, None, 73, 0, 0, "replace", 50)["waveform"].shape[-1] == 1
    # No VAE, or no window, passes the original through UNTOUCHED - not a round
    # trip of it. This is the picture-only inpaint, and it must not change.
    assert _compose(_z, None, _track, 73, 0, -1, "replace", 50) is _track
    assert _compose(_z, _AVae(), _track, 73, 0, 0, "replace", 50) is _track

    # Frames 24..48 = audio steps 40..80 = samples 32000..64000 at 800/step.
    _outw = _compose(_z, _AVae(), _track, 73, 24, 48, "replace", 0)["waveform"]
    assert _outw.shape == _track["waveform"].shape
    assert torch.equal(_outw[..., :32000], _track["waveform"][..., :32000])
    assert torch.equal(_outw[..., 64000:], _track["waveform"][..., 64000:])
    assert torch.allclose(_outw[..., 32000:64000], torch.tensor(-0.75))
    # A soundtrack shorter than the window is an error, not a silent truncation.
    try:
        _compose(_z, _AVae(), {"waveform": torch.zeros((1, 2, 1000)),
                               "sample_rate": 32000}, 73, 24, 48, "replace", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("a short soundtrack must raise")
    print("audio window + splice ok")

    print("h3.py self-check passed")
