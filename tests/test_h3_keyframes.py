"""CPU checks for MpiH3AddGuides and the per-window H3 keyframe placement in the windowed samplers.
No GPU, no weights: core's real MiniMaxH3AddGuide with a fake VAE, a fake guider that records what each
window was handed. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_h3_keyframes.py

Needs ComfyUI: set COMFY_ROOT, default G:/ComfyUi/ComfyUI.
"""
import os
import sys
import types

import torch

COMFY = os.environ.get("COMFY_ROOT", "G:/ComfyUi/ComfyUI")
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1] + ["--cpu"]

import comfy.latent_formats  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402

import sampler  # noqa: E402
from h3 import latent_frames  # noqa: E402
from h3_keyframes import (MpiH3AddGuides, MpiH3LevelGuide, even_frames, latent_pixel_starts, level_guide,  # noqa: E402
                          level_reference, timed_frames, window_keyframes)

sampler.comfy.sample.fix_empty_latent_channels = lambda patcher, samples, *a: samples
sampler.latent_preview.prepare_callback = lambda *a: (lambda *b: None)


def av(steps, audio=None):
    frames = latent_frames(steps)
    v = torch.zeros(1, 24, steps, 2, 3)
    a = torch.zeros(1, 32, 2, audio or round(frames * 5 / 3))
    return {"samples": NestedTensor((v, a))}


class FakeVae:
    """VAE.encode for a [B, H, W, C] picture: one latent step whose value is the picture's mean.
    decode gives that value back `dark` lower, as the real H3 VAE does (~0.01 = -2.5 levels)."""
    def __init__(self, dark=0.0):
        self.dark = dark

    def encode(self, pixels):
        return torch.full((1, 24, 1, 2, 3), float(pixels.mean()))

    def decode(self, latent):
        return torch.full((1, 1, 4, 4, 3), float(latent.mean()) - self.dark)


def pic(value, size=32):
    return torch.full((1, size, size, 3), value)


def kf(frame, value=1.0, steps=1):
    return {"resolved_frame_index": frame, "latent": torch.full((1, 24, steps, 2, 3), value)}


# --- spacing -------------------------------------------------------------------------------------

def test_even_frames():
    assert even_frames(0, 124) == []
    assert even_frames(1, 124) == [0]
    assert even_frames(2, 124) == [0, 123]
    assert even_frames(3, 124) == [0, 62, 123]   # the bench's A1 keys
    assert even_frames(8, 124) == [0, 18, 35, 53, 70, 88, 105, 123]
    try:
        even_frames(8, 5)
    except ValueError as exc:
        assert "8 keyframes" in str(exc)
    else:
        raise AssertionError("8 keys in 5 frames must raise, not stack two anchors on one frame")


def test_pixel_starts_match_h3():
    for steps in (2, 12, 37, 57, 97):
        assert latent_pixel_starts(steps)[-1] == latent_frames(steps)


# --- the node ------------------------------------------------------------------------------------

def run_node(**slots):
    positive = [[torch.zeros(1, 4, 8), {}]]
    return MpiH3AddGuides().doit(positive, FakeVae(), av(37), **slots)


def test_node_spaces_filled_slots_and_skips_empty():
    blank = torch.zeros(1, 1, 1, 3)  # an Mpi loader with nothing to load
    positive, frames, count = run_node(image_1=pic(0.1), image_2=blank, image_4=pic(0.4), image_8=pic(0.8))
    keys = positive[0][1]["minimax_keyframes"]
    assert (frames, count) == ("0, 62, 123", 3)
    assert [k["resolved_frame_index"] for k in keys] == [0, 62, 123]
    # slot order kept, the blank one never encoded
    assert [round(float(k["latent"].mean()), 2) for k in keys] == [0.1, 0.4, 0.8]


def test_node_with_no_pictures_is_a_passthrough():
    positive, frames, count = run_node(image_3=torch.zeros(1, 1, 1, 3))
    assert "minimax_keyframes" not in positive[0][1]
    assert (frames, count) == ("none", 0)


def test_node_keeps_keyframes_already_on_the_positive():
    # an i2v first frame upstream stays, the node's keys are appended after it
    positive = [[torch.zeros(1, 4, 8), {"minimax_keyframes": [kf(0, 9.0)]}]]
    out, _, count = MpiH3AddGuides().doit(positive, FakeVae(), av(37), image_1=pic(0.5), image_2=pic(0.5))
    assert count == 2
    assert [k["resolved_frame_index"] for k in out[0][1]["minimax_keyframes"]] == [0, 0, 123]


def test_level_guide_adds_back_the_round_trip_loss():
    out, offset = level_guide(FakeVae(dark=0.01), pic(0.5))
    assert torch.allclose(out, torch.full_like(out, 0.51)), float(out.mean())
    assert torch.allclose(offset, torch.full((3,), 0.01))
    out, _ = level_guide(FakeVae(dark=0.01), pic(1.0))   # a white picture cannot go past white
    assert float(out.max()) == 1.0


def test_level_guide_drops_alpha():
    out, _ = level_guide(FakeVae(), torch.full((1, 8, 8, 4), 0.5))
    assert out.shape[-1] == 3


def test_node_anchors_the_levelled_picture():
    positive = [[torch.zeros(1, 4, 8), {}]]
    out, _, _ = MpiH3AddGuides().doit(positive, FakeVae(dark=0.01), av(37), image_1=pic(0.4))
    # 0.41 encoded, give or take one 8-bit level: core's lanczos fit rounds through uint8
    got = float(out[0][1]["minimax_keyframes"][0]["latent"].mean())
    assert abs(got - 0.41) <= 1 / 255, got


def test_level_node_reports_its_offset():
    image, report = MpiH3LevelGuide().doit(pic(0.5), FakeVae(dark=0.01))
    assert round(float(image.mean()), 3) == 0.51
    assert report == "RGB +2.55 +2.55 +2.55", report


def test_level_reference_measures_small_and_returns_full_size():
    seen = []

    class Vae(FakeVae):
        def encode(self, pixels):
            seen.append(tuple(pixels.shape))
            return super().encode(pixels)

    big = torch.full((1, 3000, 2000, 4), 0.5)            # a full-size photo, alpha too
    out, offset = level_reference(Vae(dark=0.01), big, 704, 1216)
    assert out.shape == (1, 3000, 2000, 3)               # its own size, RGB
    assert torch.allclose(out, torch.full_like(out, 0.51)) and torch.allclose(offset, torch.full((3,), 0.01))
    h, w = seen[0][1:3]
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 704 * 1216 * 1.1, seen   # measured at ~the output's area
    out, _ = level_reference(Vae(), pic(0.5, 64), 704, 1216)
    assert seen[-1][1:3] == (64, 64)                     # a small picture is never upscaled to measure


def test_references_node_levels_every_picture_and_no_video():
    import comfy_extras.nodes_minimax_h3 as core
    from h3 import MpiH3References
    got = {}

    def execute(**kw):
        got.update(kw)
        return types.SimpleNamespace(result=("cond", "latent"))

    real = core.MiniMaxH3ReferenceToVideo.execute
    core.MiniMaxH3ReferenceToVideo.execute = execute
    try:
        clip = torch.full((5, 32, 32, 3), 0.3)
        MpiH3References().doit(None, FakeVae(dark=0.01), None, "<Picture 3> talks", 704, 1216, 124,
                               ref_image_1=torch.zeros(1, 1, 1, 3), ref_image_3=pic(0.5), ref_video_1=clip)
    finally:
        core.MiniMaxH3ReferenceToVideo.execute = real
    assert list(got["ref_images"]) == ["ref_image_0"]    # the empty slot dropped, the rest levelled
    assert round(float(got["ref_images"]["ref_image_0"].mean()), 3) == 0.51
    assert got["ref_videos"]["ref_video_0"] is clip      # a reference video is not touched
    assert got["prompt"] == "<Picture 1> talks"


def raises(fn, words):
    try:
        fn()
    except ValueError as exc:
        assert words in str(exc), str(exc)
    else:
        raise AssertionError(f"expected a ValueError naming '{words}'")


def test_timed_frames():
    assert timed_frames("", 3, 124, 5) is None and timed_frames(None, 3, 124, 5) is None
    assert timed_frames("0, 1, 5", 3, 124, 5) == [0, 25, 123]       # 1 s of 5 -> 24.6 -> 25
    assert timed_frames("0 2.5 5", 3, 124, 5) == [0, 62, 123]       # = even spacing
    assert timed_frames("0.5, 4", 2, 124, 5) == [12, 98]            # need not start on 0 or end on the length
    raises(lambda: timed_frames("0, 5", 3, 124, 5), "2 keyframe times for 3 pictures")
    raises(lambda: timed_frames("0, 6", 2, 124, 5), "outside the 5s clip")
    raises(lambda: timed_frames("-1, 5", 2, 124, 5), "outside")
    raises(lambda: timed_frames("0, 3, 2", 3, 124, 5), "rise")
    raises(lambda: timed_frames("0, 0.01", 2, 124, 5), "rise")      # both round to frame 0
    raises(lambda: timed_frames("0, 5", 2, 124, 0), "clip length")
    raises(lambda: timed_frames("0, two", 2, 124, 5), "numbers")


def test_node_places_pictures_at_their_times():
    blank = torch.zeros(1, 1, 1, 3)
    positive, frames, count = run_node(image_1=pic(0.1), image_2=blank, image_3=pic(0.3), image_4=pic(0.4),
                                       times="0, 1, 5", seconds=5.0)
    assert (frames, count) == ("0, 25, 123", 3)          # times count the FILLED pictures, blanks skipped
    keys = positive[0][1]["minimax_keyframes"]
    assert [round(float(k["latent"].mean()), 2) for k in keys] == [0.1, 0.3, 0.4]
    # blank times = even spacing, the node's default
    _, frames, _ = run_node(image_1=pic(0.1), image_2=pic(0.2), image_3=pic(0.3), times=" ", seconds=5.0)
    assert frames == "0, 62, 123"
    raises(lambda: run_node(image_1=pic(0.1), image_2=pic(0.2), times="0, 1, 5", seconds=5.0), "3 keyframe times")


# --- one window ----------------------------------------------------------------------------------

def test_window_zero_is_unchanged():
    keys = [kf(0), kf(62), kf(123)]
    kept = window_keyframes(keys, 0, 37, 37)
    assert [k["resolved_frame_index"] for k in kept] == [0, 62, 123]


def test_later_window_keeps_its_keys_on_its_own_timeline():
    full = latent_pixel_starts(97)                       # 328 frames
    start, end = 40, 77
    inside = full[45] + 2
    keys = [kf(0), kf(inside), kf(full[76]), kf(full[77]), kf(327)]
    kept = window_keyframes(keys, start, end, 97)
    win = latent_pixel_starts(end - start)
    assert [k["resolved_frame_index"] for k in kept] == [win[5] + 2, win[36]]
    assert kept[0]["resolved_frame_index"] == inside - full[start]   # on the 5-grid: a plain shift


def test_clip_anchor_keeps_overlapping_chunks():
    full = latent_pixel_starts(97)
    clip = kf(full[35], steps=12)                        # latents 35..46, chunks start at guide latent 0, 5, 10
    clip["latent"] = torch.arange(12.0).view(1, 1, 12, 1, 1).expand(1, 24, 12, 2, 3).clone()
    kept = window_keyframes([clip], 40, 77, 97)
    assert len(kept) == 1
    assert kept[0]["latent"][0, 0, :, 0, 0].tolist() == [5.0, 6, 7, 8, 9, 10, 11]
    assert kept[0]["resolved_frame_index"] == 0          # starts on the window's first frame


def test_audio_anchor_is_cropped_to_the_window_audio():
    key = {"resolved_frame_index": 30, "audio_latent": torch.arange(100.0).view(1, 1, 1, 100)}
    kept = window_keyframes([key], 10, 37, 37, audio_span=(80, 207))   # t = 50: guide 30.. lands in 80..
    assert len(kept) == 1
    assert kept[0]["audio_latent"][0, 0, 0, 0].item() == 30.0
    assert kept[0]["resolved_frame_index"] == 0.0


# --- through the samplers ------------------------------------------------------------------------

class Guider:
    """Records the keyframe frames each sampling call was handed; returns its latent unchanged."""
    def __init__(self, keys):
        self.model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(latent_format=comfy.latent_formats.MiniMaxH3AV(),
                                        process_latent_out=lambda x: x, diffusion_model=None),
            load_device="cpu")
        self.original_conds = {"positive": [{"cross_attn": None, "minimax_keyframes": keys}],
                               "negative": [{"cross_attn": None}]}
        self.seen = []

    def sample(self, noise, latent, smp, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        steps = latent.unbind()[0].shape[2]
        frames = [k["resolved_frame_index"] for k in self.original_conds["positive"][0].get("minimax_keyframes", [])]
        assert all(0 <= f < latent_frames(steps) for f in frames), (frames, steps)
        assert "minimax_keyframes" not in self.original_conds["negative"][0]
        self.seen.append(frames)
        if isinstance(smp, sampler._StartedSampler):
            smp.result = comfy.utils.pack_latents(list(latent.unbind()))[0]
        return latent


class Noise:
    seed = 7

    def generate_noise(self, latent):
        return NestedTensor([torch.zeros(t.shape) for t in latent["samples"].unbind()])


def test_windowed_sampler_places_keys_per_window():
    full = latent_pixel_starts(97)
    keys = [kf(0), kf(full[45] + 2), kf(327)]
    guider = Guider(keys)
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    _, _, info = sampler.MpiWindowedSampler().doit(
        Noise(), guider, comfy.samplers.sampler_object("euler"), sigmas, av(97), 124, 17, 5)
    assert "windows" in info, info
    assert guider.seen[0][0] == 0 and len(guider.seen) > 1
    assert all(327 not in s for s in guider.seen), guider.seen     # never at the clip's own frame number
    assert sum(len(s) for s in guider.seen) >= 3                   # every key landed somewhere
    assert guider.original_conds["positive"][0]["minimax_keyframes"] is keys  # restored after


def test_step_windowed_sampler_places_keys_per_window():
    keys = [kf(0), kf(327)]
    guider = Guider(keys)
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    sampler.MpiStepWindowedSampler().doit(
        Noise(), guider, comfy.samplers.sampler_object("euler"), sigmas, av(97), 124, 17, 5)
    assert len(guider.seen) > 2
    assert guider.seen[0] == [0]
    assert guider.seen[-1] and guider.seen[-1][-1] < 327
    assert guider.original_conds["positive"][0]["minimax_keyframes"] is keys


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)
