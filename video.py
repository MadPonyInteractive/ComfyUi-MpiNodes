from .help_funcs import video_has_audio_stream


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
