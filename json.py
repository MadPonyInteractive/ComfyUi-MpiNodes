import os
import pprint
from .help_funcs import (
    load_json_dict,
    save_json_dict,
    load_json_dict_abs,
    save_json_dict_abs,
    resolve_in_comfy_dir,
    resolve_input_file,
    list_input_files,
)


class MpiJsonLoad:
    CATEGORY = "MpiNodes/JSON"
    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("dictionary",)
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "JSON file path, relative to ComfyUI's input/ folder (an absolute path must be inside input/, output/ or temp/). Overrides the file picker when filled.",
                    },
                ),
            },
            "optional": {
                "file": (
                    list_input_files((".json",)),
                    {"tooltip": "A .json file in ComfyUI's input/ folder. Used only when path is empty."},
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, file=None):
        return True  # a file written after the list was built is still loadable

    def doit(self, path: str, file=None):
        # Contained: handing an arbitrary file's contents to the graph is an
        # arbitrary file read under the registry's policy.
        name = (path or "").strip() or file
        resolved = resolve_input_file(name)
        if not resolved:
            print(f"[MpiNodes] MpiJsonLoad: {name!r} is not inside ComfyUI's input/, output/ or temp/")
            return ({},)
        return (load_json_dict_abs(resolved),)


class MpiJsonSave:
    CATEGORY = "MpiNodes/JSON"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": (
                    "JSON",
                    {
                        "forceInput": True,
                        "tooltip": "Dictionary to save to JSON file",
                    },
                ),
                "path": (
                    "STRING",
                    {
                        "multiline": False,
                        "tooltip": "JSON file path, relative to ComfyUI's output/ folder (an absolute path must be inside output/). Missing subfolders are created.",
                    },
                ),
            }
        }

    def doit(self, json: dict, path: str):
        # Contained to output/, like core's save nodes: a free write path from
        # /prompt is an arbitrary file write under the registry's policy.
        resolved = resolve_in_comfy_dir(path, ("output",))
        if not resolved:
            raise ValueError(f"MpiJsonSave: {path!r} must be a path inside ComfyUI's output/ folder")
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        save_json_dict_abs(json, resolved)
        return (resolved,)
