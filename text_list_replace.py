class MpiTextListReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": ("STRING",),  # Accepts a list of strings
                "find_string": ("STRING", {"multiline": False}),
                "replace_string": ("STRING", {"multiline": False}),
            }
        }

    CATEGORY = "MpiNodes/TextOps"
    DESCRIPTION = "Find and replace text in each item of a text list"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_text",)
    OUTPUT_IS_LIST = (True,)  # Tells ComfyUI this is a list of STRING outputs
    INPUT_IS_LIST = True
    FUNCTION = "replace_in_list"

    def replace_in_list(self, text_list, find_string, replace_string):
        # Apply replacement
        updated = [
            t.replace(find_string[0], replace_string[0]) for t in text_list
        ]

        # Return each string as a separate output
        return (updated,)
