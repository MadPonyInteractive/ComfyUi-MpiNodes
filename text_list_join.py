class MpiTextListJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_list": ("STRING",),
                "join_text": ("STRING", {"multiline": False}),
                "mode": (["prepend", "append"],),
            }
        }

    CATEGORY = "MpiNodes/TextOps"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("updated_list",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "join_text"

    def join_text(self, text_list, join_text, mode):
        join_text = join_text[0]
        mode = mode[0]
        if mode == "prepend":
            updated = [join_text + text for text in text_list]
        else:
            updated = [text + join_text for text in text_list]
        return (updated,)
