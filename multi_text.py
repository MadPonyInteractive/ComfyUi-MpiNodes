class MpiMultiText:
    CATEGORY = "MpiNodes/Conditioning"
    DESCRIPTION = "Combine multiple text inputs into one"
    FUNCTION = "doit"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    def doit(self, **texts):
        return (" ".join(list(texts.values())),)


class MpiMultiTextX2(MpiMultiText):
    DESCRIPTION = "Combine 2 text inputs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
            }
        }


class MpiMultiTextX3(MpiMultiText):
    DESCRIPTION = "Combine 3 text inputs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
                "text_3": ("STRING", {"multiline": True}),
            }
        }


class MpiMultiTextX4(MpiMultiText):
    DESCRIPTION = "Combine 4 text inputs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
                "text_3": ("STRING", {"multiline": True}),
                "text_4": ("STRING", {"multiline": True}),
            }
        }


class MpiMultiTextX5(MpiMultiText):
    DESCRIPTION = "Combine 5 text inputs"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_1": ("STRING", {"multiline": True}),
                "text_2": ("STRING", {"multiline": True}),
                "text_3": ("STRING", {"multiline": True}),
                "text_4": ("STRING", {"multiline": True}),
                "text_5": ("STRING", {"multiline": True}),
            }
        }
