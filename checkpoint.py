class MpiToChekPoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "clip": ("CLIP", {"forceInput": True}),
                "vae": ("VAE", {"forceInput": True}),
            },
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Combines model, clip, and VAE into a checkpoint"
    FUNCTION = "doit"
    RETURN_TYPES = ("MPI_CHECKPOINT",)
    RETURN_NAMES = ("mpi_checkpoint",)

    def doit(self, **kwargs):
        return ([*kwargs.values()],)


class MpiFromCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mpi_checkpoint": ("MPI_CHECKPOINT", {"forceInput": True}),
            },
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Extracts model, clip, and VAE from a checkpoint"
    FUNCTION = "doit"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")

    def doit(self, mpi_checkpoint: list):
        print(mpi_checkpoint)
        return (*mpi_checkpoint,)
