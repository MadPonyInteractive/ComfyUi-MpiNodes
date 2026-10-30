"""CPU check for MpiWindowedSampler's automatic window (window_frames = 0). No ComfyUI, no GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_windowed_auto.py

The ceiling must land inside what MiniMax H3 refines were MEASURED to do (Cubric Studio
MPI-1350, RTX 4060 Ti 16 GB, and the same card held down to 12 GB): never above the
largest pass that still ran at full speed, and not needlessly far below it.
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

GIB = sampler.GIB
P2K = 1152 * 2048  # 2.36 M px
P1M = 768 * 1344   # 1.03 M px

# (output px, usable GiB auto saw, largest full-speed pass, smallest pass that failed or crawled)
MEASURED = [
    (P2K, 10.5, 124, 141),   # 12 GB: 141 streamed weights over PCIe, 3x slower a frame
    (P2K, 10.0, 124, 141),   # the same card on a busier desktop
    (P2K, 14.6, 192, 209),   # 16 GB: 209 ran out of memory
    (P1M, 14.7, 447, None),  # 16 GB: never failed up to 447
]


def test_never_above_what_ran_at_full_speed():
    for area, usable, ok, bad in MEASURED:
        got = sampler.auto_window_frames(area, usable * GIB)
        assert got <= ok, (area, usable, got, ok)
        if bad:
            assert got < bad, (area, usable, got, bad)
        assert type(got) is int, type(got)


def test_stops_at_the_fastest_size():
    # Past ~250 M px-frames a bigger window costs more per frame, so a card that holds
    # more (16 GB holds ~172 at 2K) still stops near 105. The old ladder gave 90.
    assert 100 <= sampler.auto_window_frames(P2K, 14.6 * GIB) <= 110
    assert 100 <= sampler.auto_window_frames(P2K, 22 * GIB) <= 110
    # A 12 GB card gets the same: the old ladder scaled it down to 64.
    assert 100 <= sampler.auto_window_frames(P2K, 10.5 * GIB) <= 110
    # The old ladder never windowed under 850 k px: a long 960x544 clip is split now.
    assert sampler.auto_window_frames(960 * 544, 48 * GIB) < 600


def test_overlap_floor_only_where_the_card_holds_it():
    # 4K on 16 GB: 250 M px-frames is 29 frames, mostly overlap; the card holds ~48.
    got = sampler.auto_window_frames(3840 * 2176, 14.6 * GIB)
    assert 45 <= got <= 51, got
    # On 12 GB the card holds ~31: memory wins over the floor.
    assert sampler.auto_window_frames(3840 * 2176, 10.5 * GIB) <= 32
    # A wider overlap raises the floor.
    assert sampler.auto_window_frames(3840 * 2176, 48 * GIB, overlap_frames=25) == 75


def test_monotonic_and_floor():
    for area in (1, 640 * 360, P1M, P2K, 2560 * 1440, 3840 * 2176, 50_000_000):
        prev = 0
        for tenth in range(0, 961):
            got = sampler.auto_window_frames(area, tenth / 10 * GIB)
            assert got >= prev, (area, tenth, got, prev)
            prev = got
        for usable in (2, 8, 16, 48):  # a bigger picture never gets a longer window
            assert sampler.auto_window_frames(area, usable * GIB) <= sampler.auto_window_frames(
                max(1, area // 2), usable * GIB)
    assert sampler.auto_window_frames(P2K, 0) == 1
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
    test_never_above_what_ran_at_full_speed()
    test_stops_at_the_fastest_size()
    test_overlap_floor_only_where_the_card_holds_it()
    test_monotonic_and_floor()
    test_usable_vram_counts_evictable_models_on_the_same_device()
    print("ok")
