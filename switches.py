from comfy_execution.graph import ExecutionBlocker  # type: ignore
from .help_funcs import comfy_paths, AlwaysEqualProxy


class MpiSwitch:
    _type: str = ""
    _type_name: str = ""

    @classmethod
    def INPUT_TYPES(cls):
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
            },
            "optional": {
                cls._type_name
                + "_1": (cls._type, {"forceInput": True, "lazy": True}),
                cls._type_name
                + "_2": (cls._type, {"forceInput": True, "lazy": True}),
                cls._type_name
                + "_3": (cls._type, {"forceInput": True, "lazy": True}),
                cls._type_name
                + "_4": (cls._type, {"forceInput": True, "lazy": True}),
                cls._type_name
                + "_5": (cls._type, {"forceInput": True, "lazy": True}),
            },
        }

    CATEGORY = "MpiNodes/Logic"
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
    RETURN_TYPES = (_type, "INT")
    RETURN_NAMES = (_type_name, "index")


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
    RETURN_TYPES = (*_types, "INT")
    RETURN_NAMES = (*_type_names, "index")


class MpiStringInvSwitch(MpiInvertedSwitch):
    _type = "STRING"
    _types = ["STRING" for i in range(1, 6)]
    _type_names = [f"str_{i}" for i in range(1, 6)]
    RETURN_TYPES = (*_types, "INT")
    RETURN_NAMES = (*_type_names, "index")
