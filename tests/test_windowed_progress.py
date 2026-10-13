"""CPU check for MpiWindowedSampler's progress: ONE bar across every window. No ComfyUI, no GPU. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_windowed_progress.py

Cubric Studio counts each restarting bar as a new stage, so a bar per window read
2/2 -> 3/3 -> 4/4 on a 3-window H3 refine (Cubric Studio MPI-1131).
"""
import io
import os
import re
import sys
import types

import torch  # noqa: F401 -- before the pack root goes on the path: its json.py shadows the stdlib torch imports
import tqdm.auto  # noqa: F401 -- same

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_previews = []  # (value, total) per UI progress update


def _prepare_callback(model, steps, x0_output_dict=None):
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0
        _previews.append((step + 1, total_steps))
    return callback


for name in ("comfy", "comfy.model_management", "comfy.sample", "comfy.utils", "comfy.nested_tensor", "latent_preview"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy.nested_tensor"].NestedTensor = object
sys.modules["latent_preview"].prepare_callback = _prepare_callback

import sampler  # noqa: E402


def test_one_bar_across_windows():
    steps, windows = 3, 3
    x0_output = {}
    err, sys.stderr = sys.stderr, io.StringIO()
    try:
        callback_for, pbar = sampler._span_progress(None, steps, windows, x0_output, False)
        for index in range(windows):
            x0_output.clear()
            cb = callback_for(index)
            for step in range(steps):
                cb(step, "x0-%d" % index, None, steps)  # the sampler passes ITS OWN total
            assert x0_output["x0"] == "x0-%d" % index
        pbar.close()
        console = sys.stderr.getvalue()
    finally:
        sys.stderr = err

    total = steps * windows
    assert _previews == [(n, total) for n in range(1, total + 1)], _previews
    # What Cubric Studio's stdout parser reads: one max, a value that never drops.
    ticks = [(int(v), int(m)) for v, m in re.findall(r"(\d+)/(\d+)\s*\[", console)]
    assert {m for _, m in ticks} == {total}, ticks
    values = [v for v, _ in ticks]
    assert values == sorted(values) and values[-1] == total, ticks


def test_disabled_pbar_still_drives_the_ui_bar():
    _previews.clear()
    callback_for, pbar = sampler._span_progress(None, 2, 2, {}, True)
    assert pbar is None
    callback_for(1)(0, None, None, 2)
    assert _previews == [(3, 4)], _previews


if __name__ == "__main__":
    test_one_bar_across_windows()
    test_disabled_pbar_still_drives_the_ui_bar()
    print("ok")
