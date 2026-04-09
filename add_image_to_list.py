class MpiAddImageToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE",),
                "image": ("IMAGE",),
            }
        }

    CATEGORY = "MpiNodes/ImgOps"
    DESCRIPTION = "Appends an image to a list of images"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("updated_list",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "append_image"

    def append_image(self, image, image_list):
        image = image[0]
        return (image_list + [image],)
