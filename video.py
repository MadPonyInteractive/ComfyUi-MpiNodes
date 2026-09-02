import os
import subprocess
import tempfile

import folder_paths  # type: ignore

from .help_funcs import (
    find_ffmpeg,
    video_has_audio_stream,
    resolve_video_path,
)


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
    # audio may be None (unconnected), an ExecutionBlocker (dead branch of a
    # router), a plain dict, or VHS's LazyAudioMap (a Mapping, not a dict).
    # Don't gate on isinstance(dict) — just try to read the waveform.
    try:
        wf = audio["waveform"]
        return wf is not None and wf.numel() > 0
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


def _probe_video_meta(ffmpeg, path):
    """Read width, height and average fps from ffmpeg's stderr stream dump.

    `ffmpeg -i <file>` with no output exits non-zero but prints the stream
    layout to stderr — we parse the Video line from that. No ffprobe needed
    (portable ComfyUI doesn't ship it). Returns (w, h, fps); fps falls back to
    24.0 if unparseable.
    """
    import re

    proc = subprocess.run([ffmpeg, "-i", path], capture_output=True)
    err = proc.stderr.decode("utf-8", "replace")

    w = h = 0
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    if m:
        w, h = int(m.group(1)), int(m.group(2))

    fps = 24.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    if m:
        fps = float(m.group(1))

    return w, h, fps


def _decode_cmd(ffmpeg, path, force_rate=0.0):
    """The rawvideo decode command, with force_rate resampling inside the single
    pass we already make.

    The `fps` FILTER, not the output `-r`. Both drop/duplicate, but -r converts to
    CFR by rounding each frame's timestamp and overshoots: measured on a 49-frame
    30 fps clip (1.633 s), `-r 24` returns 41 frames, which declared as 24 fps plays
    1.708 s -- 4.6% slow, which is the exact speed mismatch force_rate exists to
    remove. `-vf fps=24` returns 39 (1.625 s), duration preserved.
    """
    cmd = [ffmpeg, "-i", path]
    if force_rate and force_rate > 0:
        cmd += ["-vf", f"fps={force_rate:g}"]
    return cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _decode_frames(ffmpeg, path, w, h, force_rate=0.0):
    """Decode all frames to an IMAGE tensor [N, H, W, 3] in 0..1 via a raw
    rgb24 pipe. One ffmpeg pass, no intermediate files, no preview.

    force_rate > 0 resamples to that frame rate during the same pass - no
    second decode and no interpolation model."""
    import torch  # type: ignore
    import numpy as np  # type: ignore

    proc = subprocess.run(_decode_cmd(ffmpeg, path, force_rate), capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        tail = proc.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"[MpiLoadVideo] ffmpeg decode failed:\n{tail}")

    frame_bytes = w * h * 3
    n = len(proc.stdout) // frame_bytes
    arr = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8)
    arr = arr.reshape(n, h, w, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr), n


def _load_audio(ffmpeg, path):
    """Extract audio as a ComfyUI AUDIO dict, or None if the file has no audio
    track. Decodes to a temp WAV then reads it back."""
    import torch  # type: ignore

    if not video_has_audio_stream(path):
        return None

    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", path, "-vn", "-acodec", "pcm_s16le", wav_path],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(wav_path):
            return None

        import wave

        with wave.open(wav_path, "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        import numpy as np  # type: ignore

        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        pcm = pcm.reshape(-1, ch).transpose(1, 0)  # [channels, samples]
        waveform = torch.from_numpy(pcm).unsqueeze(0)  # [1, channels, samples]
        return {"waveform": waveform, "sample_rate": sr}
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


class MpiLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Video file path. Named 'string' so it matches MpiString / MpiAnyChecker outputs.",
                    },
                ),
                "block_if_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ON: empty/missing path blocks downstream execution. OFF: outputs a blank 1x1 image + silent audio so the graph continues.",
                    },
                ),
            },
            "optional": {
                "force_rate": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 240.0,
                        "step": 1.0,
                        "tooltip": "Resample the video to this frame rate while decoding. 0 = keep the source rate. ffmpeg drops/duplicates frames inside the pass that already happens, so it costs nothing and needs no interpolation model. fps, frame_count and duration are all reported at the forced rate.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "FLOAT", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "images",
        "audio",
        "fps",
        "frame_count",
        "duration",
        "width",
        "height",
        "has_audio",
    )
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = (
        "Fast, no-frills video loader: decodes frames + audio and reports "
        "source info (fps, frame_count, duration, width, height) in one "
        "ffmpeg pass. No in-graph preview and no VHS param surface, so it "
        "loads much faster than Load Video (Path). Also outputs has_audio "
        "(True when the file contains an audio track). Input is named 'string' "
        "to match MpiString / MpiAnyChecker. Empty/missing path blocks downstream "
        "unless block_if_empty is off, in which case it outputs a blank 1x1 image "
        "+ silent audio so the graph continues. force_rate resamples the video to "
        "that frame rate during the same decode pass (0 = keep the source rate); "
        "fps, frame_count and duration are then all reported at the forced rate."
    )
    FUNCTION = "load"

    def _empty(self, block_if_empty):
        from comfy_execution.graph import ExecutionBlocker  # type: ignore
        import torch  # type: ignore

        if block_if_empty:
            b = ExecutionBlocker(None)
            return (b, b, 0.0, 0, 0.0, 0, 0, False)
        image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        audio = {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}
        return (image, audio, 0.0, 0, 0.0, 0, 0, False)

    def load(self, string, block_if_empty=True, force_rate=0.0):
        path = resolve_video_path((string or "").strip())
        ffmpeg = find_ffmpeg()
        if not path or not os.path.isfile(path) or not ffmpeg:
            return self._empty(block_if_empty)

        w, h, fps = _probe_video_meta(ffmpeg, path)
        if w == 0 or h == 0:
            return self._empty(block_if_empty)

        rate = float(force_rate or 0.0)
        images, n = _decode_frames(ffmpeg, path, w, h, rate)
        if rate > 0:
            fps = rate  # report the rate the frames actually came out at
        audio = _load_audio(ffmpeg, path)
        duration = n / fps if fps else 0.0
        return (images, audio, fps, n, duration, w, h, audio is not None)


class MpiLoadAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Audio (or video) file path. Named 'string' so it matches MpiString / MpiAnyChecker outputs.",
                    },
                ),
                "block_if_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "ON: empty/missing/audio-less path blocks downstream execution. OFF: outputs silent audio so the graph continues.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = (
        "Load audio from a file path into a ComfyUI AUDIO object, like the "
        "built-in Load Audio but driven by a path string (matches MpiString / "
        "MpiAnyChecker). Works on any file ffmpeg can read, including pulling "
        "the audio track out of a video. Empty/missing/audio-less path blocks "
        "downstream unless block_if_empty is off, in which case it outputs "
        "silent audio so the graph continues."
    )
    FUNCTION = "load"

    def _empty(self, block_if_empty):
        from comfy_execution.graph import ExecutionBlocker  # type: ignore
        import torch  # type: ignore

        if block_if_empty:
            return (ExecutionBlocker(None),)
        return ({"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100},)

    def load(self, string, block_if_empty=True):
        path = resolve_video_path((string or "").strip())
        ffmpeg = find_ffmpeg()
        if not path or not os.path.isfile(path) or not ffmpeg:
            return self._empty(block_if_empty)

        audio = _load_audio(ffmpeg, path)
        if audio is None:
            return self._empty(block_if_empty)
        return (audio,)


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
                "truncate_to_audio": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "cut to audio",
                        "label_off": "keep full video",
                        "tooltip": "On = end the clip when the audio ends (shorter audio trims the video). Off = keep the full video and pad the audio with silence to match.",
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

    def doit(self, images, fps, filename_prefix="MpiVideo", use_audio=True,
             truncate_to_audio=False, audio=None):
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
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir, w, h
        )
        name = f"{filename}_{counter:05}.mp4"
        out_path = os.path.join(full_output_folder, name)

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
                cmd += ["-i", wav_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac"]
                if truncate_to_audio:
                    # Bare -shortest: end at whichever stream ends first. Short
                    # audio trims the video down to the audio length.
                    cmd += ["-shortest"]
                else:
                    # Hard-cap the output to the VIDEO's exact duration. -apad
                    # pads short audio with silence up to that -t; -t trims audio
                    # that runs longer than the video. Either way output length =
                    # video length. (apad+shortest alone let longer audio drag
                    # the clip out with a frozen last frame.)
                    vid_dur = frames.shape[0] / float(fps) if fps else 0.0
                    cmd += ["-af", "apad", "-t", f"{vid_dur:.6f}"]

            cmd += [
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                # libx264/yuv420p require even dimensions; crop off the odd
                # last row/column so 2899x806 etc. don't fail to open the encoder.
                "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-preset", "fast", "-crf", "18",
                "-movflags", "+faststart",
                out_path,
            ]

            # Stream frames to stdin instead of buffering the whole raw video as
            # one bytes blob (.tobytes() doubled a multi-GB clip in RAM). Write
            # frame-by-frame so peak overhead is one frame, not the whole video.
            frame_view = frames.numpy()
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            # Drain stderr on a thread. Otherwise ffmpeg — chattier once an audio
            # input is added — fills the ~64KB stderr pipe, blocks, stops reading
            # stdin, and we deadlock writing frames (node hangs forever). Reading
            # stderr only after the write loop is the bug; read it concurrently.
            import threading
            stderr_chunks = []
            drain = threading.Thread(
                target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True
            )
            drain.start()

            try:
                for f in frame_view:
                    proc.stdin.write(f.tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass  # ffmpeg died early; its stderr below has the real reason
            rc = proc.wait()
            drain.join()
            stderr = stderr_chunks[0] if stderr_chunks else b""
            if rc != 0:
                tail = stderr.decode("utf-8", "replace")[-2000:]
                raise RuntimeError(f"[MpiSaveVideo] ffmpeg failed:\n{tail}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

        return {"ui": {"videos": [
            {"filename": name, "subfolder": subfolder, "type": "output", "format": "video/mp4"}
        ]}}


class MpiAudioRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The clip's full soundtrack."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01, "tooltip": "Frame rate the start/end indices are counted in. Wire the loader's own fps - a guessed rate slides the audio against the picture instead of failing."}),
                "start": ("INT", {"default": 0, "min": -0xFFFFFFFFFFFFFFFF, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Start FRAME (inclusive). Negative counts from the end. Same convention as MpiListRange, so both nodes take the same two numbers."}),
                "end": ("INT", {"default": -1, "min": -0xFFFFFFFFFFFFFFFF, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "End FRAME (inclusive). -1 is the last frame. Same convention as MpiListRange."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = (
        "Cut a soundtrack to a FRAME range, using the same inclusive start/end that "
        "MpiListRange takes, so one pair of numbers windows the picture and the sound "
        "together. Exists for windowed work on an audio-video model: cutting the frames "
        "with MpiListRange and leaving the audio whole hands the model a soundtrack that "
        "does not line up with the picture, and nothing downstream can notice - the "
        "tensors are all valid, the result is just conditioned on the wrong moment."
    )
    FUNCTION = "doit"

    def doit(self, audio, fps, start, end):
        waveform = audio["waveform"]
        rate = audio["sample_rate"]
        total = waveform.shape[-1]
        frames = max(1, round(total / rate * fps))

        s = start + frames if start < 0 else start
        e = end + frames if end < 0 else end
        s = max(0, min(s, frames - 1))
        e = max(0, min(e, frames - 1))
        if e < s:
            return ({"waveform": waveform[..., :0], "sample_rate": rate},)

        # Clamped rather than trusted: frames is derived from the waveform length,
        # so the last frame's end can round a sample or two past the real tail.
        i0 = min(total, round(s / fps * rate))
        i1 = min(total, round((e + 1) / fps * rate))
        return ({"waveform": waveform[..., i0:i1], "sample_rate": rate},)


class MpiAudioSplice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "The clip's FULL soundtrack - the one MpiAudioRange was cut from."}),
                "patch": ("AUDIO", {"tooltip": "The window to write back in, e.g. the audio output of MpiH3DecodeAV. Resampled to the full track's rate if it does not already match."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0, "step": 0.01, "tooltip": "Frame rate the start index is counted in. Wire the loader's own fps - a guessed rate slides the patch against the picture instead of failing."}),
                "start": ("INT", {"default": 0, "min": -0xFFFFFFFFFFFFFFFF, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Start FRAME (inclusive) the patch lands on. Negative counts from the end. Same convention as MpiListRange and MpiAudioRange, so the number that CUT the window writes it back."}),
                "crossfade": ("INT", {"default": 50, "min": 0, "max": 2000, "tooltip": "Crossfade at each end of the patch, in milliseconds, ramped INSIDE the patch so the rest of the track is untouched. A hard splice clicks, and a click is the one artefact an audio edit cannot hide."}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    CATEGORY = "MpiNodes/Video"
    DESCRIPTION = (
        "Write a run of audio back into a longer soundtrack at a FRAME offset - the "
        "inverse of MpiAudioRange, and the sound half of what MpiImageSplice does for "
        "picture. Windowed work on an audio-video model cuts both streams to a window, "
        "generates, and has to splice both back; without this the picture returns "
        "full-length while regenerated audio stays window-length, which reads as a "
        "video whose sound stops early rather than as a wiring mistake. A patch "
        "running past the end of the track raises rather than landing silently short."
    )
    FUNCTION = "doit"

    def doit(self, audio, patch, fps, start, crossfade):
        import torch

        from .h3 import splice_audio

        waveform = audio["waveform"]
        rate = audio["sample_rate"]
        wave_p = patch["waveform"]
        if patch["sample_rate"] != rate:
            import torchaudio

            wave_p = torchaudio.functional.resample(wave_p, patch["sample_rate"], rate)

        total = waveform.shape[-1]
        frames = max(1, round(total / rate * fps))
        s = start + frames if start < 0 else start
        s = max(0, min(s, frames - 1))
        i0 = min(total, round(s / fps * rate))
        i1 = i0 + wave_p.shape[-1]
        if i1 > total:
            raise ValueError(
                f"A {wave_p.shape[-1]}-sample patch placed at frame {s} runs "
                f"{i1 - total} samples past the end of a {total}-sample track. "
                "Either start names a different window than the patch came from, "
                "or the patch came from a different clip."
            )

        # splice_audio indexes both tensors in the SAME coordinates, so the patch is
        # laid into a full-length buffer rather than the offset being threaded
        # through it -- one allocation, and the crossfade stays the tested one.
        if wave_p.shape[1] == 1 and waveform.shape[1] > 1:
            wave_p = wave_p.expand(-1, waveform.shape[1], -1)
        placed = torch.zeros_like(waveform[:1])
        placed[..., i0:i1] = wave_p[:1].to(placed)
        spliced = splice_audio(
            waveform[:1], placed, i0, i1, int(rate * crossfade / 1000))
        return ({"waveform": spliced, "sample_rate": rate},)
