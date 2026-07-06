import os
import subprocess
import tempfile

import folder_paths  # type: ignore

from .help_funcs import find_ffmpeg, video_has_audio_stream


class MpiHasAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to the video file (same string fed to VHS_LoadVideoPath). Absolute path or a bare basename resolved against ComfyUI's input dir.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_audio",)
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = (
        "Check if a video file contains an audio stream via ffprobe. "
        "Outputs True if an audio track exists, False otherwise. "
        "Reads the file (not an AUDIO object) to avoid false positives from "
        "lazy audio maps. Defaults to False if ffprobe is missing or errors. "
        "Use to gate an audio wire into CreateVideo so video-only clips "
        "don't crash the workflow."
    )
    FUNCTION = "doit"

    def doit(self, video_path: str):
        return (video_has_audio_stream(video_path),)


def _audio_has_samples(audio):
    """True only if `audio` carries a real, non-empty waveform.

    A connected AUDIO wire is not proof of sound: source nodes that expose an
    AUDIO socket but produce nothing hand down a None/zero-length/all-silent
    waveform. Encoding that stalls the video-only path (and drove the
    invert-switch + double-node workaround). Gate on actual samples instead.
    """
    # audio may be None (unconnected) or an ExecutionBlocker (dead branch of a
    # router like MpiIfElseInverted). Only a real AUDIO dict has "waveform".
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return False
    wf = audio["waveform"]
    try:
        if wf.numel() == 0:
            return False
        # all-zero waveform == silence from a placeholder source -> treat as none
        return bool(wf.abs().max().item() > 0)
    except Exception:
        return False


def _write_wav(waveform, sample_rate, path):
    """Write a ComfyUI AUDIO waveform tensor [B, C, samples] to a WAV file.

    Uses soundfile if available, else stdlib `wave` (16-bit PCM). No new
    dependency required — one of these is always around on a ComfyUI engine.
    """
    import torch  # type: ignore

    # Take first batch item -> [channels, samples] -> [samples, channels]
    wf = waveform[0].transpose(0, 1).contiguous().cpu()

    try:
        import soundfile as sf  # type: ignore

        sf.write(path, wf.numpy(), int(sample_rate))
        return
    except Exception:
        pass

    import wave

    pcm = (wf.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


class MpiSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01},
                ),
                "filename_prefix": ("STRING", {"default": "MpiVideo"}),
            },
            "optional": {
                "use_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "with audio",
                        "label_off": "video only",
                        "tooltip": "Off = ignore the audio input and encode video-only. Lets you toggle audio with one boolean — wire audio straight in, no if/else router needed.",
                    },
                ),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = (
        "Fast save-video node with no in-graph preview and optional audio. "
        "Encodes an IMAGE frame batch (+ optional AUDIO) to a single .mp4 in "
        "one libx264 pass, on the engine — much faster than CreateVideo+"
        "SaveVideo, and remote gens transfer only the final mp4, not the raw "
        "frames. Toggle audio with the use_audio boolean (wire audio straight "
        "in, no if/else router). GPU-agnostic (no nvenc)."
    )
    FUNCTION = "doit"

    def doit(self, images, fps, filename_prefix="MpiVideo", use_audio=True, audio=None):
        import torch  # type: ignore

        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "[MpiSaveVideo] no ffmpeg found (imageio-ffmpeg bundle / "
                "VHS_FORCE_FFMPEG_PATH / PATH all failed)."
            )

        h, w = images.shape[1], images.shape[2]
        frames = images.clamp(0.0, 1.0).mul(255.0).round().to("cpu", torch.uint8).contiguous()

        out_dir = folder_paths.get_output_directory()
        full_prefix, _, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w, h
        )
        name = f"{os.path.basename(full_prefix)}_{counter:05}.mp4"
        out_path = os.path.join(os.path.dirname(full_prefix), name)

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        ]

        wav_path = None
        try:
            if use_audio and _audio_has_samples(audio):
                wav_path = tempfile.mktemp(suffix=".wav")
                _write_wav(audio["waveform"], audio["sample_rate"], wav_path)
                cmd += ["-i", wav_path, "-map", "0:v", "-map", "1:a",
                        "-c:a", "aac", "-shortest"]

            cmd += [
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "18",
                "-movflags", "+faststart",
                out_path,
            ]

            proc = subprocess.run(cmd, input=frames.numpy().tobytes(), capture_output=True)
            if proc.returncode != 0:
                tail = proc.stderr.decode("utf-8", "replace")[-2000:]
                raise RuntimeError(f"[MpiSaveVideo] ffmpeg failed:\n{tail}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        return {"ui": {"videos": [
            {"filename": name, "subfolder": subfolder, "type": "output", "format": "video/mp4"}
        ]}}
