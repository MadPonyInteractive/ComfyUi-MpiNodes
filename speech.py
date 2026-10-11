"""Speech to text (Cubric Studio MPI-1071): Qwen3-ASR on the engine's own transformers (5.13 ships
`qwen3_asr`), so the pack gains no pip requirement. Weights are a Hugging Face folder under
models/speech_to_text/<name>/ (config.json, model.safetensors, tokenizer, processor config)."""
import os

import torch
import torchaudio

import comfy.model_management as mm  # type: ignore
import folder_paths  # type: ignore

folder_paths.add_model_folder_path("speech_to_text", os.path.join(folder_paths.models_dir, "speech_to_text"))

_ASR = {}  # ponytail: one model kept on the CPU between runs (~1.6 GB RAM for 0.6B); drop it if RAM bites


def _asr_models():
    names = set()
    for root in folder_paths.get_folder_paths("speech_to_text"):
        if os.path.isdir(root):
            names.update(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "config.json")))
    return sorted(names)


def _asr_path(name):
    for root in folder_paths.get_folder_paths("speech_to_text"):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise FileNotFoundError(f"speech_to_text model not found: {name}")


def _load_asr(name):
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    path = _asr_path(name)
    if _ASR.get("path") != path:
        _ASR.clear()
        device = mm.get_torch_device()
        dtype = torch.bfloat16 if mm.should_use_bf16(device) else torch.float16 if device.type == "cuda" else torch.float32
        model = Qwen3ASRForConditionalGeneration.from_pretrained(path, dtype=dtype)
        model.train(False)
        _ASR.update(path=path, model=model, processor=AutoProcessor.from_pretrained(path))
    return _ASR["model"], _ASR["processor"]


class MpiSpeechToText:
    """AUDIO in, the words spoken out. `context` is free text the model reads first (names, terms
    it should spell your way); empty is plain transcription in whatever language is spoken."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (_asr_models(),),
            },
            "optional": {
                "context": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "doit"
    CATEGORY = "Mpi/Audio"

    def doit(self, audio, model, context=""):
        asr, processor = _load_asr(model)
        rate = processor.feature_extractor.sampling_rate
        wave = audio["waveform"][0].float().mean(0)  # first item, down-mixed to mono
        if audio["sample_rate"] != rate:
            wave = torchaudio.functional.resample(wave, audio["sample_rate"], rate)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": context.strip()}]},
            {"role": "user", "content": [{"type": "audio", "audio": wave.numpy()}]},
        ]
        inputs = processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        device = mm.get_torch_device()
        mm.free_memory(mm.module_size(asr) * 2, device)  # ponytail: weights + activations guess
        asr.to(device)
        try:
            inputs = inputs.to(device, asr.dtype)
            with torch.inference_mode():
                ids = asr.generate(**inputs, max_new_tokens=1024)
            text = processor.decode(ids[0, inputs["input_ids"].shape[1]:], return_format="transcription_only")
        finally:
            asr.to("cpu")
            mm.soft_empty_cache()
        return (text.strip() if isinstance(text, str) else str(text).strip(),)
