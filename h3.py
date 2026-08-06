"""MiniMax H3 frame timing.

H3 only generates frame counts on a 17k+5 grid at 24 fps, so most durations are
not reachable: asking for 2 s gets you 2.33 s (56 frames). The core H3 nodes snap
`length` up internally and never tell you what they picked, which leaves the graph
(and the UI in front of it) guessing. This node does the same arithmetic up front
and reports the truth.
"""

FPS = 24.0
_GRID = 17
_OFFSET = 5
# From the core node's own tooltip (comfy_extras/nodes_minimax_h3.py): default 124,
# "trained range is ~124-362". Outside it the model still runs, untested.
TRAINED_MIN = 124
TRAINED_MAX = 362


def snap_h3_frames(frames: int) -> int:
    """Nearest valid H3 frame count (n % 17 == 5), minimum 5.

    Core snaps UP, which maximises the error — 4 s asks for 96 frames and gets
    107 (4.46 s) when 90 (3.75 s) is closer. Nearest is never worse. The result
    is already valid, so the core node's own snap leaves it alone.
    """
    k = round((frames - _OFFSET) / _GRID)
    return max(_OFFSET, k * _GRID + _OFFSET)


class MpiH3Length:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.2,
                        "max": 150.0,
                        "step": 0.1,
                        "tooltip": "Wanted duration. H3 can only land on a 17k+5 frame grid at 24 fps, so the `seconds` output is what you will actually get.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("frames", "seconds", "in_trained_range")
    CATEGORY = "MpiNodes/Utils"
    DESCRIPTION = "Convert a wanted duration into a valid MiniMax H3 frame count. H3 only generates n % 17 == 5 frames at 24 fps, so exact whole seconds are mostly impossible - 8 s is the shortest one that lands. Wire `frames` into the H3 node's `length`, and show `seconds` in the UI instead of what was asked for. `in_trained_range` is false outside 124-362 frames (~5.2-15.1 s), where the model still runs but was not trained."
    FUNCTION = "doit"

    def doit(self, seconds: float):
        frames = snap_h3_frames(round(seconds * FPS))
        return (frames, frames / FPS, TRAINED_MIN <= frames <= TRAINED_MAX)
