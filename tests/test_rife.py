"""CPU checks for rife/ (MPI-1342): MpiRifeVFI and the RIFE 4.7 IFNet, with a randomly initialised
network (no weights, no download). Where the original comfyui-frame-interpolation pack is on disk
the port is compared to it on fixed inputs and the same random state dict. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_rife.py
"""
import importlib.util
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = COMFY + "/custom_nodes/comfyui-frame-interpolation"
sys.path.insert(0, COMFY)

# CPU for everything, whatever the box has (the nodes bind get_torch_device at import time)
import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")


def _load_pkg(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


port = _load_pkg("mpi_rife_port", os.path.join(ROOT, "rife"))
nodes = sys.modules["mpi_rife_port.nodes"]
arch = sys.modules["mpi_rife_port.arch"]

orig_arch = orig_node = None
if os.path.isdir(ORIG):
    sys.path.insert(0, ORIG)
    try:
        import vfi_models.rife as orig_node  # noqa: E402
        import vfi_models.rife.rife_arch as orig_arch  # noqa: E402
        orig_arch.device = torch.device("cpu")
    finally:
        sys.path.remove(ORIG)


def _frames(n, h=24, w=40, seed=0):
    torch.manual_seed(seed)
    return torch.rand(n, h, w, 3)


def _net(seed=0):
    torch.manual_seed(seed)
    net = arch.IFNet().eval()
    for p in net.parameters():  # random init is small; widen it so the flow/mask are not ~constant
        p.data.mul_(2.0)
    return net


def test_mapping_and_inputs():
    assert set(port.NODE_CLASS_MAPPINGS) == {"MpiRifeVFI"}
    assert port.NODE_DISPLAY_NAME_MAPPINGS["MpiRifeVFI"] == "Mpi RIFE VFI"
    inp = nodes.MpiRifeVFI.INPUT_TYPES()["required"]
    assert inp["ckpt_name"][0] == ["rife47.pth"]
    # every value video_interpolate.json sets
    assert inp["scale_factor"][0].count(1.0) == 1 and 1 in inp["scale_factor"][0]
    assert "float32" in inp["dtype"][0]
    assert set(inp) == {"ckpt_name", "frames", "clear_cache_after_n_frames", "multiplier", "fast_mode",
                        "ensemble", "scale_factor", "dtype", "torch_compile", "batch_size"}
    if orig_node:
        o = orig_node.RIFE_VFI.INPUT_TYPES()["required"]
        for k in inp:
            if k != "ckpt_name":
                assert inp[k] == o[k], k


def test_task_plan():
    assert nodes.plan_tasks(3, 1) == []
    t = nodes.plan_tasks(3, 4)
    assert t == [(0, .25), (0, .5), (0, .75), (1, .25), (1, .5), (1, .75)]
    assert nodes.build_scale_list(1.0) == [8.0, 4.0, 2.0, 1.0]
    assert nodes.build_scale_list(0.5) == [16.0, 8.0, 4.0, 2.0]


def test_net_matches_original():
    net = _net()
    a, b = _frames(1, 24, 40, 1).permute(0, 3, 1, 2), _frames(1, 24, 40, 2).permute(0, 3, 1, 2)
    t = torch.tensor([0.5]).view(1, 1, 1, 1)
    with torch.no_grad():
        out = net(a, b, t, [8, 4, 2, 1], False, True)
        assert out.shape == a.shape and 0 <= float(out.min()) and float(out.max()) <= 1
        if orig_arch:
            o = orig_arch.IFNet(arch_ver="4.7").eval()
            o.load_state_dict(net.state_dict())  # same key names
            for args in ((False, True, False), (True, True, False), (False, False, True)):
                ours = net(a, b, t, [8, 4, 2, 1], *args)
                theirs = o(a, b, t, [8, 4, 2, 1], *args)
                assert torch.equal(ours, theirs), args


def test_node_matches_original():
    net = _net(3)
    with tempfile.TemporaryDirectory() as d:
        wpath = os.path.join(d, "rife47.pth")
        torch.save(net.state_dict(), wpath)
        nodes.rife_weight_dirs = lambda: [d]
        nodes._model_cache.clear()
        frames = _frames(3)
        node = nodes.MpiRifeVFI()
        base = dict(ckpt_name="rife47.pth", clear_cache_after_n_frames=2, multiplier=3, fast_mode=False,
                    ensemble=True, scale_factor=1, dtype="float32", torch_compile=False, batch_size=1)
        out = node.vfi(frames=frames, **base)[0]
        assert out.shape == (3 + 2 * 2, 24, 40, 3) and out.dtype == torch.float32
        assert torch.equal(out[0], frames[0]) and torch.equal(out[3], frames[1]) and torch.equal(out[-1], frames[-1])
        # batching must not change the result
        nodes._model_cache.clear()
        out_b = node.vfi(frames=frames, **{**base, "batch_size": 4})[0]
        assert torch.allclose(out, out_b, atol=1e-5)
        # a missing weight names the folders instead of downloading
        nodes.rife_weight_dirs = lambda: [os.path.join(d, "nope")]
        nodes._model_cache.clear()
        try:
            node.vfi(frames=frames, **base)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as e:
            assert "nope" in str(e)
        if orig_node:
            orig_node.load_file_from_github_release = lambda model_type, name: wpath
            orig_node._model_cache.clear()
            theirs = orig_node.RIFE_VFI().vfi(frames=frames, **base)[0]
            assert torch.equal(out, theirs), "port differs from original node"
            base2 = {**base, "multiplier": 2, "scale_factor": 0.5, "fast_mode": True, "ensemble": False}
            nodes.rife_weight_dirs = lambda: [d]
            nodes._model_cache.clear()
            orig_node._model_cache.clear()
            assert torch.equal(node.vfi(frames=frames, **base2)[0], orig_node.RIFE_VFI().vfi(frames=frames, **base2)[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all rife tests passed" + ("" if orig_node else " (original pack absent: equivalence skipped)"))
