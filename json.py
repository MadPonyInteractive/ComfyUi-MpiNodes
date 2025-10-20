import pprint
from .help_funcs import (
    load_json_dict,
    save_json_dict,
    load_json_dict_abs,
    save_json_dict_abs,
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
                        "tooltip": "Full path to json file to load",
                    },
                ),
            }
        }

    def doit(self, path: str):
        return (load_json_dict_abs(path),)


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
                        # "default": "full_path\\to\\new_file.json",
                        "tooltip": "Full path to json file to save",
                    },
                ),
            }
        }

    def doit(self, json: dict, path: str):
        save_json_dict_abs(json, path)
        return (path,)
