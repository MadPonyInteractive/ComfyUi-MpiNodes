import os
import unicodedata


class MpiBatchTextReplace:
    CATEGORY = "MpiNodes/TextOps"
    DESCRIPTION = "Batch find and replace text in all .txt files within a folder"
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
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        count = 0
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".txt"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

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

        return (f"Processed {count} file(s)", output_folder)


# NODE_CLASS_MAPPINGS = {"MpiBatchTextReplace": MpiBatchTextReplace}

# NODE_DISPLAY_NAME_MAPPINGS = {"MpiBatchTextReplace": "Mpi Batch Text Replace"}
