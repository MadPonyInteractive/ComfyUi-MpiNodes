"""CPU check for MpiWindowedSampler's automatic window (window_frames = 0). No ComfyUI, no GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_windowed_auto.py

On a 16 GB card the automatic window must be IDENTICAL to the "Window Frames" MpiMath
ladder Cubric Studio's H3 graphs carried before it (Cubric Studio MPI-1350). The ladder
string is evaluated here through the same `safe_math` MpiMath runs, so the comparison is
against what the graphs actually computed, not a retyping of it.
"""
import os
import sys
import types

import torch  # noqa: F401 -- before the pack root goes on the path: its json.py shadows the stdlib torch imports
import tqdm.auto  # noqa: F401 -- same

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for name in ("comfy", "comfy.model_management", "comfy.sample", "comfy.utils", "comfy.nested_tensor",
             "latent_preview", "folder_paths"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy.nested_tensor"].NestedTensor = object
sys.modules["latent_preview"].prepare_callback = None
mm = sys.modules["comfy.model_management"]
sys.modules["comfy"].model_management = mm

import sampler  # noqa: E402
from help_funcs import safe_math  # noqa: E402

GIB = sampler.GIB
LADDER = "100000 if a * b < 850000 else (243 if a * b < 1100000 else (90 if a * b < 6000000 else 39))"

# Every bucket edge from both sides, plus the sizes the H3 graphs really sample (each
# side a multiple of 32): 640x360, 960x544, 1280x720, 1920x1088, 2560x1440, 3840x2176.
SIZES = [(1, 1), (640, 360), (960, 544), (1280, 720), (1920, 1088), (2560, 1440), (3840, 2176)]
AREAS = [w * h for w, h in SIZES] + [849999, 850000, 850001, 1099999, 1100000, 1100001,
                                     5999999, 6000000, 6000001, 50_000_000]

# What a 16 GB card shows the sampler: 15.99 GiB bare (Linux Pod), down to ~14 GiB with a
# busy Windows desktop.
USABLE_16GB = [14 * GIB, 14.5 * GIB, 15 * GIB, 15.5 * GIB, 15.92 * GIB, 15.99 * GIB, 16 * GIB]


def ladder(area):
    return safe_math(LADDER, {"a": area, "b": 1, "c": 0})


def test_16gb_matches_the_ladder_exactly():
    for usable in USABLE_16GB:
        for area in AREAS:
            got = sampler.auto_window_frames(area, usable)
            assert got == ladder(area), (usable / GIB, area, got, ladder(area))
            assert type(got) is int, type(got)


def test_ladder_helper_is_the_graph_expression():
    for area in AREAS + list(range(849_990, 850_010)) + list(range(1_099_990, 1_100_010)) \
            + list(range(5_999_990, 6_000_010)):
        assert sampler.ladder_16gb(area) == ladder(area), area


def test_scales_with_memory_outside_the_band():
    area = 2560 * 1440  # the 90-frame bucket
    assert sampler.auto_window_frames(area, 24 * GIB) == int(90 * 24 / 16)
    assert sampler.auto_window_frames(area, 80 * GIB) == int(90 * 80 / 16)
    assert sampler.auto_window_frames(area, 12 * GIB) == int(90 * 12 / 14)
    # No step at the band edges: just outside is within a frame of just inside.
    for edge in sampler.REF_USABLE:
        for bump in (-1, 1):
            near = sampler.auto_window_frames(area, edge + bump * 1024 ** 2)
            assert abs(near - 90) <= 1, (edge / GIB, bump, near)
    # Monotonic in memory, at every bucket.
    for a in AREAS:
        prev = 0
        for gib in range(2, 97):
            got = sampler.auto_window_frames(a, gib * GIB)
            assert got >= prev, (a, gib, got, prev)
            prev = got


def test_never_window_bucket():
    small = 640 * 360
    assert sampler.auto_window_frames(small, 48 * GIB) == 100000
    # A smaller card cannot promise a whole clip: it takes the 243 bucket's ceiling, scaled.
    assert sampler.auto_window_frames(small, 7 * GIB) == int(243 * 7 / 14)
    assert sampler.auto_window_frames(1, 0) == 1


class _Loaded:
    def __init__(self, device, size):
        self.device, self.size = device, size

    def model_loaded_memory(self):
        return self.size


def test_usable_vram_counts_evictable_models_on_the_same_device():
    mm.get_free_memory = lambda dev: 3 * GIB
    mm.current_loaded_models = [_Loaded("cuda:0", 9 * GIB), _Loaded("cuda:0", 2 * GIB),
                                _Loaded("cuda:1", 40 * GIB)]
    assert sampler.usable_vram("cuda:0") == 14 * GIB


if __name__ == "__main__":
    test_16gb_matches_the_ladder_exactly()
    test_ladder_helper_is_the_graph_expression()
    test_scales_with_memory_outside_the_band()
    test_never_window_bucket()
    test_usable_vram_counts_evictable_models_on_the_same_device()
    print("ok")
