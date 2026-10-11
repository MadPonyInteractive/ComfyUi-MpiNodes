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


FRAME_S = 0.03     # loudness is measured per 30 ms frame
PAUSE_S = 0.3      # a quiet run this long is a pause; shorter dips are part of the talk
MIN_S = 1.0        # a piece is never cut shorter than this
MAX_S = 20.0       # a longer stretch is cut at its longest pause; shorter ones stay whole
HARD_S = 30.0      # no pause by then: the first pause after MAX_S, else a cut at MAX_S
CUE_CHARS = 84     # a longer piece becomes several subtitles
BLIP_S = 0.12      # a loud stretch shorter than this, alone between pauses, is ignored


def _segments(wave, rate):
    """[(start_s, end_s, spans)] of the speech in a mono waveform. Up to MAX_S is one piece; longer
    audio is cut at the longest pause in each MAX_S window (cutting at every pause broke sentences
    and lost words at the joins: "remembers you. Give us" for "remembers to give us"). Each piece is
    trimmed to its talk; `spans` are its talking stretches, pauses left out, which time the
    subtitles. An all-quiet piece is dropped: the model invents words for silence.
    ponytail: loudness only, no voice detector, no word alignment; add one if subtitles drift."""
    hop = max(1, int(rate * FRAME_S))
    n = wave.shape[-1] // hop
    if n == 0:
        return []
    db = 20 * torch.log10(wave[: n * hop].reshape(n, hop).pow(2).mean(1).sqrt() + 1e-9)
    floor, peak = torch.quantile(db, 0.1).item(), torch.quantile(db, 0.9).item()
    # A quarter of the way from the quiet floor to the loud level; never below -50 dBFS (silence).
    level = floor + 0.25 * (peak - floor) if peak - floor > 6 else -50.0
    loud = (db > max(level, -50.0)).tolist()
    pauses, run = [], 0   # (middle_s, length_s)
    for i, on in enumerate(loud + [True]):
        if not on:
            run += 1
            continue
        if run * FRAME_S >= PAUSE_S:
            pauses.append(((i - run / 2) * FRAME_S, run * FRAME_S))
        run = 0
    end = n * FRAME_S
    pieces, start = [], 0.0
    while end - start > 1e-6:
        if end - start <= MAX_S:
            cut = end
        else:
            inside = [p for p in pauses if start + MIN_S < p[0] <= start + MAX_S]
            later = [p for p in pauses if start + MAX_S < p[0] <= start + HARD_S]
            cut = max(inside, key=lambda p: p[1])[0] if inside else later[0][0] if later else start + MAX_S
        spans = []
        for i in range(int(round(start / FRAME_S)), min(n, int(round(cut / FRAME_S)))):
            if not loud[i]:
                continue
            if spans and i * FRAME_S - spans[-1][1] < PAUSE_S:
                spans[-1][1] = (i + 1) * FRAME_S
            else:
                spans.append([i * FRAME_S, (i + 1) * FRAME_S])
        # A blip under BLIP_S is a click, not a word: a mic-stop click put a subtitle 1.2 s early.
        spans = [tuple(s) for s in spans if s[1] - s[0] >= BLIP_S]
        if spans:
            pieces.append((spans[0][0], spans[-1][1], spans))
        start = cut
    return pieces


def _srt(pieces):
    """SRT from [(start_s, end_s, text, spans)]. A piece over CUE_CHARS splits at spaces into
    several cues, its talking time (`spans`, pauses left out) shared by characters, so a cue
    starts after a pause rather than inside it."""
    def stamp(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"

    def wall(spans, x, starting):
        """The clock time `x` seconds of talk into `spans`; a cue start lands after a pause."""
        for a, b in spans:
            if x < b - a or (not starting and x <= b - a):
                return a + x
            x -= b - a
        return spans[-1][1]

    cues = []
    for start, end, text, spans in pieces:
        lines, line = [], ""
        for word in text.split():
            if line and len(line) + 1 + len(word) > CUE_CHARS:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        spans = spans or [(start, end)]
        talk, total, done = sum(b - a for a, b in spans), sum(len(x) for x in lines), 0
        for x in lines:
            cues.append((wall(spans, talk * done / total, True), wall(spans, talk * (done + len(x)) / total, False), x))
            done += len(x)
    return "".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{t}\n\n" for i, (a, b, t) in enumerate(cues, 1))


class MpiSpeechToText:
    """AUDIO in, the words spoken out, and the same words as SRT subtitles. Up to 20 s is read in
    one pass; longer audio is cut at its longest pauses into pieces of up to 20 s, which keeps a long
    file from running past the model's output limit and gives the subtitle times. `context` is free
    text the model reads first (names, terms it should spell your way); empty is plain
    transcription in whatever language is spoken."""

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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "srt")
    FUNCTION = "doit"
    CATEGORY = "Mpi/Audio"

    def doit(self, audio, model, context=""):
        asr, processor = _load_asr(model)
        rate = processor.feature_extractor.sampling_rate
        wave = audio["waveform"][0].float().mean(0)  # first item, down-mixed to mono
        if audio["sample_rate"] != rate:
            wave = torchaudio.functional.resample(wave, audio["sample_rate"], rate)
        device = mm.get_torch_device()
        mm.free_memory(mm.module_size(asr) * 2, device)  # ponytail: weights + activations guess
        asr.to(device)
        pieces = []
        try:
            for start, end, spans in _segments(wave, rate):
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": context.strip()}]},
                    {"role": "user", "content": [{"type": "audio", "audio": wave[int(start * rate):int(end * rate)].numpy()}]},
                ]
                inputs = processor.apply_chat_template(
                    [messages], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = inputs.to(device, asr.dtype)
                with torch.inference_mode():
                    ids = asr.generate(**inputs, max_new_tokens=1024)
                text = processor.decode(ids[0, inputs["input_ids"].shape[1]:], return_format="transcription_only")
                text = (text if isinstance(text, str) else str(text)).strip()
                if text:
                    pieces.append((start, end, text, spans))
        finally:
            asr.to("cpu")
            mm.soft_empty_cache()
        return (" ".join(p[2] for p in pieces), _srt(pieces))
