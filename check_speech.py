"""Run MpiSpeechToText on real clips with no ComfyUI server (Cubric Studio MPI-1071).

    <ComfyUI>/python_embeded/python.exe check_speech.py <models_dir> <model_name> <clip.wav>... [--context "Cubric Studio"] [--srt] [--whole]
    <ComfyUI>/python_embeded/python.exe check_speech.py --selftest

<models_dir> holds speech_to_text/<model_name>/ (the Hugging Face folder). Prints each clip's text
and the time it took. Uses the GPU when there is one; lease it first on a shared machine.
--srt also prints the subtitles; --whole transcribes each clip in one pass (the node before pieces),
to compare. --selftest checks the pause cutting and the SRT text on synthetic audio, no model.
"""
import os, sys, time, types, wave

import numpy as np
import torch

PACK = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
context = ""
if "--context" in args:
    i = args.index("--context")
    context = args[i + 1]
    del args[i:i + 2]
flags = {a for a in args if a.startswith("--")}
args = [a for a in args if not a.startswith("--")]
models_dir, name, *clips = args if "--selftest" not in flags else (PACK, "", [])

# Same synthetic-package import as check_audio_range.py, with just enough of ComfyUI stubbed.
_pkg = types.ModuleType("mpinodes")
_pkg.__path__ = [PACK]
sys.modules["mpinodes"] = _pkg
fp = types.ModuleType("folder_paths")
fp.models_dir = models_dir
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda kind: [os.path.join(models_dir, kind)]
sys.modules["folder_paths"] = fp
mm = types.ModuleType("comfy.model_management")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mm.get_torch_device = lambda: dev
mm.should_use_bf16 = lambda d: d.type == "cuda" and torch.cuda.is_bf16_supported()
mm.module_size = lambda m: 0
mm.free_memory = lambda *a, **k: None
mm.soft_empty_cache = lambda: torch.cuda.empty_cache() if torch.cuda.is_available() else None
comfy = types.ModuleType("comfy")
comfy.model_management = mm
sys.modules["comfy"] = comfy
sys.modules["comfy.model_management"] = mm

import importlib
speech = importlib.import_module("mpinodes.speech")

if "--selftest" in flags:
    rate = 16000
    tone = lambda s: 0.3 * torch.sin(torch.arange(int(s * rate)) / rate * 2 * torch.pi * 220)
    hush = lambda s: torch.zeros(int(s * rate)) + 1e-4 * torch.randn(int(s * rate))
    near = lambda x, y: abs(x - y) < 0.06
    # Under MAX_S stays ONE piece: 3 s talk, 0.5 s pause, 3 s talk, 0.2 s dip, 4 s talk, quiet tail.
    w = torch.cat([hush(0.5), tone(3), hush(0.5), tone(3), hush(0.2), tone(4), hush(1)])
    (a, b, spans), = speech._segments(w, rate)
    assert near(a, 0.5) and near(b, 11.2), (a, b)                 # trimmed to the talk
    assert len(spans) == 2 and near(spans[0][1], 3.5) and near(spans[1][0], 4.0), spans  # dip merged
    # Over MAX_S: cut at the LONGEST pause in the window (1.0 s at ~13.7 s), not the last one.
    w = torch.cat([tone(8), hush(0.4), tone(5), hush(1.0), tone(10)])
    segs = speech._segments(w, rate)
    assert len(segs) == 2 and near(segs[0][1], 13.4) and near(segs[1][0], 14.4), segs
    assert speech._segments(hush(5), rate) == [], 'all quiet = nothing to transcribe'
    # A 0.06 s click 1 s before the talk does not start the piece.
    (a, b, spans), = speech._segments(torch.cat([hush(1), tone(0.06), hush(1), tone(3), hush(1)]), rate)
    assert near(a, 2.06), (a, spans)
    long = speech._segments(tone(50), rate)                       # no pause at all: hard cuts at MAX_S
    assert len(long) == 3 and all(b - a <= speech.MAX_S + 0.05 for a, b, _ in long), long
    srt = speech._srt([(1.0, 3.5, 'Hello there.', None)])
    assert srt == "1\n00:00:01,000 --> 00:00:03,500\nHello there.\n\n", srt
    speech.CUE_CHARS = 3                                           # two cues, talk time halved
    srt = speech._srt([(60.0, 70.0, "aaa bbb", [(60.0, 62.0), (68.0, 70.0)])])
    # The second cue starts after the pause (68 s), not inside it (62 s).
    assert srt == "1\n00:01:00,000 --> 00:01:02,000\naaa\n\n2\n00:01:08,000 --> 00:01:10,000\nbbb\n\n", srt
    assert speech._srt([]) == ''
    print("selftest ok")
    sys.exit(0)

assert name in speech._asr_models(), f"{name} not in {speech._asr_models()}"
node = speech.MpiSpeechToText()
if "--whole" in flags:
    speech._segments = lambda w, r: [(0.0, w.shape[-1] / r, None)]


def load_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, rate, width = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
        assert width == 2, "16-bit PCM only"
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(-1, ch).T / 32768.0
    return {"waveform": torch.from_numpy(data).float()[None], "sample_rate": rate}


for clip in clips:
    t0 = time.time()
    text, srt = node.doit(load_wav(clip), name, context)
    print(f"[{time.time() - t0:.1f} s] {os.path.basename(clip)}: {text}")
    if "--srt" in flags:
        print(srt)
