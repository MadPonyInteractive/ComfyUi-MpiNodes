from comfy_execution.graph import ExecutionBlocker  # type: ignore
from .help_funcs import comfy_paths, AlwaysEqualProxy, values_equal


class MpiSwitch:
    _type: str = ""
    _type_name: str = ""
    _count: int = 5

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": cls._count,
                        "tooltip": f"Selection from 1 to {cls._count} \nOutput index is also 1 to {cls._count}",
                    },
                ),
            },
            "optional": {
                f"{cls._type_name}_{i}": (
                    cls._type,
                    {"forceInput": True, "lazy": True},
                )
                for i in range(1, cls._count + 1)
            },
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Select one of up to 5 inputs based on selection index"
    FUNCTION = "use_selected"

    def check_lazy_status(self, select: int, **kwargs):
        lst = list(kwargs.keys())
        if lst and len(lst) >= select:
            return [lst[select - 1]]
        return []

    def use_selected(self, select: int, **kwargs):
        lst = list(kwargs.values())
        if lst and len(lst) >= select:
            return (lst[select - 1], select)
        return (ExecutionBlocker(None), select)


class MpiLoraSwitch(MpiSwitch):
    _type = AlwaysEqualProxy("*")
    _type_name = "lora_name"
    DESCRIPTION = "Select one of up to 5 LoRA files"

    @classmethod
    def INPUT_TYPES(cls):
        file_list = comfy_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "select": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "tooltip": "Selection from 1 to 5 \nOutput index is also 1 to 5",
                    },
                ),
                # },
                # "optional": {
                cls._type_name + "_1": (file_list, {"lazy": True}),
                cls._type_name + "_2": (file_list, {"lazy": True}),
                cls._type_name + "_3": (file_list, {"lazy": True}),
                cls._type_name + "_4": (file_list, {"lazy": True}),
                cls._type_name + "_5": (file_list, {"lazy": True}),
            },
        }

    RETURN_TYPES = (_type, "INT")
    RETURN_NAMES = (_type_name, "index")


class MpiAnySwitch(MpiSwitch):
    _type = AlwaysEqualProxy("*")
    _type_name = "any"
    DESCRIPTION = "Select one of up to 5 inputs of any type"
    RETURN_TYPES = (_type, "INT")
    RETURN_NAMES = (_type_name, "index")


class MpiAnySwitch10(MpiAnySwitch):
    _count = 10
    DESCRIPTION = "Select one of up to 10 inputs of any type"


class MpiInvertedSwitch:
    _type = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (cls._type, {}),
                "select": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 5,
                        "tooltip": "Selection from 1 to 5 \nOutput index is also 1 to 5",
                    },
                ),
            },
        }

    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = "Route input to one of up to 5 outputs based on selection"
    FUNCTION = "use_selected"

    # @classmethod
    # def VALIDATE_INPUTS(cls, input_types):
    #     return True

    def use_selected(self, input, select: int):
        ret = []
        for i in range(1, 6):
            if i == select:
                ret.append(input)
            else:
                ret.append(ExecutionBlocker(None))
        return tuple(ret)


class MpiAnyInvSwitch(MpiInvertedSwitch):
    _type = AlwaysEqualProxy("*")
    _types = [AlwaysEqualProxy("*") for i in range(1, 6)]
    _type_names = [f"any_{i}" for i in range(1, 6)]
    DESCRIPTION = "Route input of any type to one of up to 5 outputs"
    RETURN_TYPES = (*_types, "INT")
    RETURN_NAMES = (*_type_names, "index")


class MpiStringInvSwitch(MpiInvertedSwitch):
    _type = "STRING"
    _types = ["STRING" for i in range(1, 6)]
    _type_names = [f"str_{i}" for i in range(1, 6)]
    DESCRIPTION = "Route string input to one of up to 5 outputs"
    RETURN_TYPES = (*_types, "INT")
    RETURN_NAMES = (*_type_names, "index")


class MpiPacker:
    _count = 5

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                f"any_{i}": (
                    AlwaysEqualProxy("*"),
                    {
                        "forceInput": True,
                        "tooltip": "Any value. Unconnected slots come out of Mpi Unpacker blocked.",
                    },
                )
                for i in range(1, cls._count + 1)
            },
        }

    RETURN_TYPES = ("MPI_PACK",)
    RETURN_NAMES = ("pack",)
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "Bundle up to 5 values of any type into one MPI_PACK wire. Mpi Unpacker "
        "gives them back in the same slot order. A pack is just a list, so a pack "
        "can go into another packer's slot to nest them."
    )
    FUNCTION = "pack"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def pack(self, **kwargs):
        # Read slots by name rather than kwargs order — the order ComfyUI hands
        # them over follows the prompt JSON, not INPUT_TYPES.
        return ([kwargs.get(f"any_{i}") for i in range(1, self._count + 1)],)


class MpiUnpacker:
    _count = 5

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pack": ("MPI_PACK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),) * 5
    RETURN_NAMES = ("any_1", "any_2", "any_3", "any_4", "any_5")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "Unpack an MPI_PACK back into 5 outputs, in the same slot order they "
        "were packed. Slots that were empty block execution downstream."
    )
    FUNCTION = "unpack"

    def unpack(self, pack):
        # A 10-slot pack in a 5-slot unpacker silently loses the tail, which is
        # a miserable thing to debug. Only warn when real values are dropped.
        if any(v is not None for v in pack[self._count :]):
            print(
                f"[MpiNodes] Mpi Unpacker: pack holds {len(pack)} slots but this "
                f"node only has {self._count} outputs — slots "
                f"{self._count + 1}-{len(pack)} were dropped. Use Mpi Unpacker 10."
            )
        return tuple(
            pack[i] if i < len(pack) and pack[i] is not None else ExecutionBlocker(None)
            for i in range(self._count)
        )


class MpiPacker10(MpiPacker):
    _count = 10
    DESCRIPTION = "Same as Mpi Packer with 10 slots instead of 5"


class MpiUnpacker10(MpiUnpacker):
    _count = 10
    RETURN_TYPES = (AlwaysEqualProxy("*"),) * 10
    RETURN_NAMES = tuple(f"any_{i}" for i in range(1, 11))
    DESCRIPTION = "Same as Mpi Unpacker with 10 outputs instead of 5"


class MpiComparePacks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("MPI_PACK", {"forceInput": True}),
                "b": ("MPI_PACK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "INT")
    RETURN_NAMES = ("equal", "first_diff")
    CATEGORY = "MpiNodes/Logic"
    DESCRIPTION = (
        "Compare two packs slot by slot. Outputs equal (true only if every slot "
        "matches) and first_diff, the 1-based slot of the first mismatch (0 when "
        "equal). Images and other tensors compare by content, and nested packs "
        "compare all the way down."
    )
    FUNCTION = "compare"

    def compare(self, a, b):
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b) or not values_equal(a[i], b[i]):
                return (False, i + 1)
        return (True, 0)
