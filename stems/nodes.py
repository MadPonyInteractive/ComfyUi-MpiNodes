"""Stem separation for Cubric Studio's Stems Flow (MPI-1342).

Ported from audio-separation-nodes-comfyui (src/separation.py + ensure_stereo from
src/utils.py), MIT licence, Copyright (c) 2025 Christian Byrne. See
stems/LICENSE-audio-separation. Only AudioSeparation is ported; behaviour is unchanged.
Credit for the chunked separation:
https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html

Needs only torch and torchaudio (both owned by the engine): no librosa (upstream imports it
for its tempo nodes, which Separation never touches) and no moviepy.

The Hybrid Demucs v3 weight (hdemucs_high_trained.pt, ~320 MB) is fetched by torchaudio into
the torch hub cache on first run, exactly as upstream: same file, same place.
"""
from __future__ import annotations

from zipfile import BadZipFile

import comfy.model_management
import torch
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade, Resample


def ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    """Stereo from [channels, frames] or [batch, channels, frames]: mono is duplicated,
    multi-channel is downmixed (first two channels averaged) and duplicated."""
    if audio.ndim not in (2, 3):
        raise ValueError("Audio input must have 2 or 3 dimensions: [channels, frames] or [batch, channels, frames].")

    is_batched = audio.ndim == 3
    channels_dim = 1 if is_batched else 0

    if audio.shape[channels_dim] == 2:
        return audio
    elif audio.shape[channels_dim] == 1:
        return audio.repeat(1, 2, 1) if is_batched else audio.repeat(2, 1)

    audio = audio.narrow(channels_dim, 0, 2).mean(dim=channels_dim, keepdim=True)
    return audio.repeat(1, 2, 1) if is_batched else audio.repeat(2, 1)


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device = None,
    chunk_fade_shape: str = "linear",
) -> torch.Tensor:
    """Apply `model` to `mix` [batch, channels, length] chunk by chunk, fade each chunk's
    tail and overlap-add the chunks (the first chunk has no fade-in, the last no fade-out).
    Returns [batch, n_sources, channels, length]."""
    device = mix.device if device is None else torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape=chunk_fade_shape)

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


class MpiAudioSeparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunk_fade_shape": (
                    [
                        "linear",
                        "half_sine",
                        "logarithmic",
                        "exponential",
                    ],
                    {
                        "default": "linear",
                        "tooltip": "Audio is split into segments (chunks) with overlapping areas to ensure smooth transitions. This setting controls the fade effect at these overlaps. Choose Linear for even fading, Half-Sine for a smooth curve, Logarithmic for a quick fade out and slow fade in, or Exponential for a slow fade out and quick fade in.",  # noqa: E501
                    },
                ),
                "chunk_length": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "tooltip": "The length of each segment (chunk) in seconds. Longer chunks may require more memory and MIGHT produce better results.",  # noqa: E501
                    },
                ),
                "chunk_overlap": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "tooltip": "The overlap between each segment (chunk) in seconds. A higher overlap may be necessary if chunks are too short or the audio changes rapidly.",  # noqa: E501
                    },
                ),
            },
        }

    FUNCTION = "main"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("Bass", "Drums", "Other", "Vocals")
    CATEGORY = "MpiNodes/Audio"
    DESCRIPTION = "Separate audio into four sources: bass, drums, other, and vocals."

    bundle = HDEMUCS_HIGH_MUSDB_PLUS

    def _load_model(self):
        try:
            return self.bundle.get_model()
        except (BadZipFile, RuntimeError) as exc:
            raise RuntimeError(
                "Failed to load the Hybrid Demucs model — the downloaded checkpoint "
                "appears to be corrupted. Delete the cached model file and restart "
                "ComfyUI to trigger a fresh download. The cached file is typically "
                "located in your torch hub cache directory (~/.cache/torch/hub/checkpoints/)."
            ) from exc

    def main(self, audio, chunk_fade_shape="linear", chunk_length=10.0, chunk_overlap=0.1):
        device = comfy.model_management.get_torch_device()
        waveform = audio["waveform"]
        waveform = waveform.squeeze(0).to(device)
        input_sample_rate = audio["sample_rate"]

        model = self._load_model()
        model = model.to(device)
        self.model_sample_rate = self.bundle.sample_rate

        waveform = ensure_stereo(waveform)
        waveform = waveform.float()

        if input_sample_rate != self.model_sample_rate:
            resample = Resample(input_sample_rate, self.model_sample_rate).to(device)
            waveform = resample(waveform)

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # Zs

        sources = separate_sources(
            model,
            waveform[None],
            self.model_sample_rate,
            segment=chunk_length,
            overlap=chunk_overlap,
            device=device,
            chunk_fade_shape=chunk_fade_shape,
        )[0]
        sources = sources * ref.std() + ref.mean()
        sources_list = model.sources
        sources = list(sources)

        return self.sources_to_tuple(dict(zip(sources_list, sources)))

    def sources_to_tuple(self, sources):
        output_order = ["bass", "drums", "other", "vocals"]
        outputs = []
        for source in output_order:
            if source not in sources:
                raise ValueError(f"Missing source {source} in the output")
            outputs.append(
                {
                    "waveform": sources[source].cpu().unsqueeze(0),
                    "sample_rate": self.model_sample_rate,
                }
            )
        return tuple(outputs)


NODE_CLASS_MAPPINGS = {
    "MpiAudioSeparation": MpiAudioSeparation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiAudioSeparation": "Mpi Audio Separation",
}
