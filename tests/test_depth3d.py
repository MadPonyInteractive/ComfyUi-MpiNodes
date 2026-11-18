"""CPU checks for depth3d/stream.py (MpiDepthVideo's windowed Video Depth Anything) with a fake model.
No ComfyUI, no GPU, no weights. Run with the engine's python:

    G:/ComfyUi/python_embeded/python.exe tests/test_depth3d.py

The reference below is upstream's offline infer_video_depth loop (video_depth.py @ 4f5ae23), kept
as written, with the model call swapped for the same fake. DepthWindows must give the same frames.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depth3d.stream import (INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP, DepthWindows,  # noqa: E402
                            net_size, prepare, scale_and_shift, to_grey)


class FakeModel:
    """Depth = channel mean of the input, scaled and shifted per call, so alignment has work to do."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):  # [1, T, 3, h, w] -> [T, h, w]
        self.calls += 1
        s, b = 1.0 + 0.37 * self.calls, 0.11 * self.calls
        return (x[0].mean(1) * s + b + 3.0).numpy().astype(np.float32)


def upstream(frames_in, size):
    """video_depth.py's infer_video_depth + alignment, model swapped for FakeModel."""
    model = FakeModel()
    frame_list = list(prepare(frames_in, size).unbind(0))
    frame_step = INFER_LEN - OVERLAP
    org_video_len = len(frame_list)
    append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
    frame_list = frame_list + [frame_list[-1].clone()] * append_frame_len
    depth_list = []
    pre_input = None
    for frame_id in range(0, org_video_len, frame_step):
        cur_input = torch.stack(frame_list[frame_id:frame_id + INFER_LEN]).unsqueeze(0)
        if pre_input is not None:
            cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]
        depth_list += list(model(cur_input))
        pre_input = cur_input

    depth_list_aligned = []
    ref_align = []
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]
    for frame_id in range(0, len(depth_list), INFER_LEN):
        if len(depth_list_aligned) == 0:
            depth_list_aligned += depth_list[:INFER_LEN]
            for kf_id in kf_align_list:
                ref_align.append(depth_list[frame_id + kf_id])
        else:
            curr_align = [depth_list[frame_id + i] for i in range(len(kf_align_list))]
            scale, shift = scale_and_shift(np.concatenate(curr_align), np.concatenate(ref_align))
            pre_depth_list = depth_list_aligned[-INTERP_LEN:]
            post_depth_list = depth_list[frame_id + align_len:frame_id + OVERLAP]
            for i in range(len(post_depth_list)):
                post_depth_list[i] = post_depth_list[i] * scale + shift
                post_depth_list[i][post_depth_list[i] < 0] = 0
            w = [i / (INTERP_LEN - 1) for i in range(INTERP_LEN)]
            depth_list_aligned[-INTERP_LEN:] = [pre_depth_list[i] * (1 - w[i]) + post_depth_list[i] * w[i]
                                                for i in range(INTERP_LEN)]
            for i in range(OVERLAP, INFER_LEN):
                new_depth = depth_list[frame_id + i] * scale + shift
                new_depth[new_depth < 0] = 0
                depth_list_aligned.append(new_depth)
            ref_align = ref_align[:1]
            for kf_id in kf_align_list[1:]:
                new_depth = depth_list[frame_id + kf_id] * scale + shift
                new_depth[new_depth < 0] = 0
                ref_align.append(new_depth)
    return np.stack(depth_list_aligned[:org_video_len])


def streamed(frames_in, size, chunk):
    dw = DepthWindows(FakeModel(), size)
    out = []
    for i in range(0, len(frames_in), chunk):
        out += dw.push(frames_in[i:i + chunk])
    out += dw.finish()
    return np.stack(out)


def test_stream_matches_upstream():
    torch.manual_seed(0)
    size = (28, 42)
    for n in (1, 5, 21, 22, 31, 32, 33, 54, 55, 76, 100):
        frames = torch.rand(n, 20, 30, 3)
        ref = upstream(frames, size)
        for chunk in (1, 7, 24, 1000):
            got = streamed(frames, size, chunk)
            assert got.shape == ref.shape == (n, *size), (n, chunk, got.shape, ref.shape)
            assert np.allclose(got, ref, atol=1e-5), (n, chunk, float(np.abs(got - ref).max()))


def test_net_size():
    assert net_size(1080, 1920) == (518, 924)  # 16:9
    assert net_size(1920, 1080) == (924, 518)  # 9:16
    assert net_size(1024, 1024) == (518, 518)
    h, w = net_size(720, 1440)  # 2:1 shrinks the short side below 518, still /14
    assert h % 14 == 0 and w % 14 == 0 and h < 518


def test_to_grey():
    d = np.array([[1.0, 2.0], [3.0, 5.0]])
    g = to_grey(d, 1.0, 5.0)
    assert g.min() == 0.0 and g.max() == 1.0 and g[1, 1] == 1.0  # nearest (largest) is white
    assert np.all(to_grey(d, 2.0, 2.0) >= 0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
