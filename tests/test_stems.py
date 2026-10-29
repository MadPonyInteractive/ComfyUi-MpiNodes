"""CPU checks for stems/ (MPI-1342): MpiAudioSeparation chunking, overlap-add and resampling, with
a fake model standing in for Hybrid Demucs (no weights, no download). Where the original pack is on
disk the port is compared to it on fixed inputs. Run with:

    G:/ComfyUi/python_embeded/python.exe tests/test_stems.py
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "G:/ComfyUi/ComfyUI"
ORIG = COMFY + "/custom_nodes/audio-separation-nodes-comfyui"
sys.path.insert(0, COMFY)


def _load_pkg(name, path, init="__init__.py"):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, init), submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


port = _load_pkg("mpi_stems_port", os.path.join(ROOT, "stems"))
nodes = sys.modules["mpi_stems_port.nodes"]
orig = None
if os.path.isdir(ORIG):
    _load_pkg("orig_audiosep", ORIG)
    orig = sys.modules["orig_audiosep.src.separation"]

# CPU for both, whatever the box has
import comfy.model_management as mm  # noqa: E402
mm.get_torch_device = lambda: torch.device("cpu")

SR = 8000


class FakeModel(torch.nn.Module):
    """Four 'stems': the chunk scaled by 1, .5, .25, .125 (so overlap-add must rebuild each)."""
    sources = ["drums", "bass", "other", "vocals"]  # Demucs order, NOT the output order

    def forward(self, x):
        return torch.stack([x * s for s in (1.0, 0.5, 0.25, 0.125)], dim=1)


class FakeBundle:
    sample_rate = SR

    def get_model(self):
        return FakeModel()


def _audio(seconds, sr, channels=2):
    torch.manual_seed(0)
    t = torch.arange(int(seconds * sr)) / sr
    w = torch.sin(2 * torch.pi * 220 * t) + 0.3 * torch.randn(len(t))
    return {"waveform": w.repeat(channels, 1)[None] * 0.5, "sample_rate": sr}


def test_mapping():
    assert set(port.NODE_CLASS_MAPPINGS) == {"MpiAudioSeparation"}
    assert port.NODE_DISPLAY_NAME_MAPPINGS["MpiAudioSeparation"] == "Mpi Audio Separation"
    cls = port.NODE_CLASS_MAPPINGS["MpiAudioSeparation"]
    inp = cls.INPUT_TYPES()
    assert inp["optional"]["chunk_fade_shape"][0] == ["linear", "half_sine", "logarithmic", "exponential"]
    assert cls.RETURN_NAMES == ("Bass", "Drums", "Other", "Vocals")
    if orig:
        o = orig.AudioSeparation
        assert inp == o.INPUT_TYPES() and cls.RETURN_NAMES == o.RETURN_NAMES and cls.RETURN_TYPES == o.RETURN_TYPES


def test_overlap_add_rebuilds_signal():
    # a linear-faded overlap-add of an identity-like model gives the input back (every fade is
    # complementary), for any chunk length that does not divide the signal
    mix = torch.randn(1, 2, 8000 * 7 + 123)
    for shape in ("linear", "half_sine"):
        out = nodes.separate_sources(FakeModel(), mix, SR, segment=2.0, overlap=0.1, chunk_fade_shape=shape)
        assert out.shape == (1, 4, 2, mix.shape[-1])
        tol = 1e-4 if shape == "linear" else 0.5  # half_sine is not power-complementary-to-1 by design
        err = (out[:, 0] - mix).abs().max().item()
        if shape == "linear":
            assert err < tol, err
        assert torch.allclose(out[:, 1], out[:, 0] * 0.5)


def test_short_input_single_chunk():
    mix = torch.randn(1, 2, 1000)
    out = nodes.separate_sources(FakeModel(), mix, SR, segment=10.0, overlap=0.1)
    # first chunk: no fade-in. Upstream quirk kept: when the first chunk is also the only one its
    # tail still gets the overlap fade-out (the fade-out is only cleared AFTER that chunk)
    assert torch.allclose(out[:, 0, :, :150], mix[:, :, :150], atol=1e-6)
    assert out[:, 0, :, -1].abs().max() < mix[:, :, -1].abs().max()


def test_ensure_stereo():
    assert nodes.ensure_stereo(torch.ones(1, 5)).shape == (2, 5)
    assert nodes.ensure_stereo(torch.ones(1, 1, 5)).shape == (1, 2, 5)
    six = torch.arange(6.0).view(6, 1).repeat(1, 4)
    st = nodes.ensure_stereo(six)
    assert st.shape == (2, 4) and torch.equal(st[0], st[1]) and float(st[0, 0]) == 0.5
    try:
        nodes.ensure_stereo(torch.ones(5))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_node_output_order_rate_and_scale():
    node = port.NODE_CLASS_MAPPINGS["MpiAudioSeparation"]()
    node.bundle = FakeBundle()
    audio = _audio(3.0, 16000)  # model wants 8000: resampling path
    bass, drums, other, vocals = node.main(audio, "half_sine", 1.0, 0.1)
    for o in (bass, drums, other, vocals):
        assert o["sample_rate"] == SR and o["waveform"].shape[:2] == (1, 2)
    assert bass["waveform"].shape[-1] == 3 * SR
    # scale factors 0.5 / 1 / .25 / .125 for bass / drums / other / vocals survive the z-score round trip
    r = drums["waveform"].abs().mean() / bass["waveform"].abs().mean()
    assert abs(float(r) - 2.0) < 0.05, float(r)
    # mono in -> stereo out
    mono = _audio(1.0, SR, channels=1)
    out = node.main(mono)
    assert out[0]["waveform"].shape[1] == 2


def test_matches_original():
    if orig is None:
        return
    for sr, shape, ln, ov in ((16000, "linear", 1.0, 0.1), (8000, "half_sine", 0.7, 0.25), (22050, "exponential", 1.5, 0.0)):
        audio = _audio(2.5, sr)
        a = port.NODE_CLASS_MAPPINGS["MpiAudioSeparation"]()
        a.bundle = FakeBundle()
        b = orig.AudioSeparation()
        saved = orig.HDEMUCS_HIGH_MUSDB_PLUS
        orig.HDEMUCS_HIGH_MUSDB_PLUS = FakeBundle()
        try:
            ra, rb = a.main(audio, shape, ln, ov), b.main(audio, shape, ln, ov)
        finally:
            orig.HDEMUCS_HIGH_MUSDB_PLUS = saved
        for x, y in zip(ra, rb):
            assert x["sample_rate"] == y["sample_rate"] and torch.equal(x["waveform"], y["waveform"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all stems tests passed" + ("" if orig else " (original pack absent: equivalence skipped)"))
