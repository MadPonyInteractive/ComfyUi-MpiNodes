"""Run MpiSpeechToText on real clips with no ComfyUI server (Cubric Studio MPI-1071).

    <ComfyUI>/python_embeded/python.exe check_speech.py <models_dir> <model_name> <clip.wav>... [--context "Cubric Studio"]

<models_dir> holds speech_to_text/<model_name>/ (the Hugging Face folder). Prints each clip's text
and the time it took. Uses the GPU when there is one; lease it first on a shared machine.
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
models_dir, name, *clips = args

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
assert name in speech._asr_models(), f"{name} not in {speech._asr_models()}"
node = speech.MpiSpeechToText()


def load_wav(path):
    with wave.open(path, "rb") as w:
        n, ch, rate, width = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
        assert width == 2, "16-bit PCM only"
        data = np.frombuffer(w.readframes(n), dtype="<i2").reshape(-1, ch).T / 32768.0
    return {"waveform": torch.from_numpy(data).float()[None], "sample_rate": rate}


for clip in clips:
    t0 = time.time()
    (text,) = node.doit(load_wav(clip), name, context)
    print(f"[{time.time() - t0:.1f} s] {os.path.basename(clip)}: {text}")
