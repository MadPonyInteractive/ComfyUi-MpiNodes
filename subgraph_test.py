# from ..comfy_types import ComfyNodeABC, IO
# from ...comfy_execution.graph_utils import GraphBuilder
# from ... import folder_paths
from comfy_execution.graph_utils import GraphBuilder  # type:ignore
import folder_paths  # type:ignore
from .help_funcs import aspect_ratio


class MpiSubGraphTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path1": (
                    folder_paths.get_filename_list("checkpoints"),
                    {},
                ),
                "checkpoint_path2": (
                    folder_paths.get_filename_list("checkpoints"),
                    {},
                ),
                "ratio": (
                    "FLOAT",
                    {
                        "default": 1.00,
                        "min": 0.00,
                        "max": 1.00,
                    },
                ),
            }
        }

    CATEGORY = "MpiNodes/Dev"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "check"

    def check(self, width: int, height: int):
        return (aspect_ratio(width, height),)

    def load_and_merge_checkpoints(
        self, checkpoint_path1, checkpoint_path2, ratio
    ):
        graph = GraphBuilder()
        checkpoint_node1 = graph.node(
            "CheckpointLoaderSimple", checkpoint_path=checkpoint_path1
        )
        checkpoint_node2 = graph.node(
            "CheckpointLoaderSimple", checkpoint_path=checkpoint_path2
        )
        merge_model_node = graph.node(
            "ModelMergeSimple",
            model1=checkpoint_node1.out(0),
            model2=checkpoint_node2.out(0),
            ratio=ratio,
        )
        merge_clip_node = graph.node(
            "ClipMergeSimple",
            clip1=checkpoint_node1.out(1),
            clip2=checkpoint_node2.out(1),
            ratio=ratio,
        )
        return {
            # Returning (MODEL, CLIP, VAE) outputs
            "result": (
                merge_model_node.out(0),
                merge_clip_node.out(0),
                checkpoint_node1.out(2),
            ),
            "expand": graph.finalize(),
        }
