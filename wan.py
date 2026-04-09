class MpiWanFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "INT",
                    {
                        "default": 16,
                        "step": 4,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "step": 0.1,
                        "min": 1.0,
                        "max": 120.0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("frames", "seconds", "fps")
    CATEGORY = "MpiNodes/Wan"
    DESCRIPTION = "Calculate duration from frame count and FPS"
    FUNCTION = "doit"

    def doit(self, frames: int, fps: float):
        return (frames, (frames - 1) / fps, fps)


class MpiWanSeconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": (
                    "INT",
                    {
                        "default": 1,
                        "step": 1,
                        "min": 1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 16.0,
                        "step": 0.1,
                        "min": 1.0,
                        "max": 120.0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("frames", "seconds", "fps")
    CATEGORY = "MpiNodes/Wan"
    DESCRIPTION = "Calculate frame count from duration and FPS"
    FUNCTION = "doit"

    def doit(self, seconds: int, fps: float):
        return (int((seconds * fps) + 1), seconds, fps)
