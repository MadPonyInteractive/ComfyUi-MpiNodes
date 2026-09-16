import os
import unicodedata

from .help_funcs import resolve_in_comfy_dir


class MpiBatchTextReplace:
    CATEGORY = "MpiNodes/TextOps"
    DESCRIPTION = (
        "Batch find and replace text in all .txt files within a folder. "
        "input_folder is read under ComfyUI's input/ (or output/) folder and "
        "output_folder is written under output/; a path outside them is refused."
    )
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "output_folder")
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"multiline": False}),
                "output_folder": ("STRING", {"multiline": False}),
                "find_string": ("STRING", {"multiline": False}),
                "replace_string": ("STRING", {"multiline": False}),
            }
        }

    def process(
        self, input_folder, output_folder, find_string, replace_string
    ):
        # Contained to ComfyUI's own folders: a free output path plus makedirs
        # and a write is an arbitrary file write from /prompt (registry verdict
        # on 1.2.4 / 1.2.5).
        src = resolve_in_comfy_dir(input_folder, ("input", "output"))
        if not src or not os.path.isdir(src):
            raise ValueError(
                f"input_folder must be a folder inside ComfyUI's input/ or output/: {input_folder}"
            )
        dst = resolve_in_comfy_dir(output_folder, ("output",))
        if not dst:
            raise ValueError(f"output_folder must be inside ComfyUI's output/: {output_folder}")
        os.makedirs(dst, exist_ok=True)

        count = 0
        for filename in os.listdir(src):
            if filename.lower().endswith(".txt"):
                input_path = os.path.join(src, filename)
                output_path = os.path.join(dst, filename)

                # Try reading with UTF-8, fallback to cp1252
                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(input_path, "r", encoding="cp1252") as f:
                        content = f.read()

                # Normalize legacy characters
                content = unicodedata.normalize("NFKD", content)

                updated_content = content.replace(find_string, replace_string)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)

                count += 1

        return (f"Processed {count} file(s)", dst)


# NODE_CLASS_MAPPINGS = {"MpiBatchTextReplace": MpiBatchTextReplace}

# NODE_DISPLAY_NAME_MAPPINGS = {"MpiBatchTextReplace": "Mpi Batch Text Replace"}
