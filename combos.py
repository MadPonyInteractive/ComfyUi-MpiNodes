import pprint
from .help_funcs import (
    load_combo,
    load_json_dict,
    save_json_dict,
    load_json_dict_abs,
    save_json_dict_abs,
)


class MpiComboMaker:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Create, edit, or delete custom combo options"
    RETURN_TYPES = ("STRING", "JSON")
    RETURN_NAMES = ("combos", "json")
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combo_name": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "My new combo",
                    },
                ),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                    },
                ),
                "operation": (["create/edit", "delete", "load/save"],),
            },
            "optional": {
                "json": (
                    "JSON",
                    {
                        "forceInput": True,
                        "tooltip": "Use for loading previous files upon an update or re-install of the custom node \nIt will overwritte all combos with the inputed value",
                    },
                ),
            },
        }

    def doit(
        self, combo_name: str, options: str, operation: str, json: dict = {}
    ):
        file = "web/user/combos.json"

        combos: dict = json or load_json_dict(file)

        if operation == "delete":
            if combos.get(combo_name):
                del combos[combo_name]
        elif operation == "create/edit":
            combos[combo_name] = options.splitlines()

        save_json_dict(combos, file)
        return (",".join(combos.keys()), combos)


class MpiCombo:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from custom combo options and return selected value and index"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected", "index")
    FUNCTION = "use_selected"

    @classmethod
    def INPUT_TYPES(cls):
        cls.preset = load_json_dict("web/user/combos.json")
        cls.keys = list(cls.preset.keys())
        cls.all_options = list(
            set(item for sublist in cls.preset.values() for item in sublist)
        )

        return {
            "required": {
                "combo_name": (
                    cls.keys,
                    {"tooltip": "Use 'Mpi Combo Maker' to create combos!"},
                ),
                "options": (cls.all_options,),
            }
        }

    def use_selected(self, combo_name, options):
        for i, option in enumerate(self.preset.get(combo_name, [])):
            if option == options:
                return (option, i + 1)
        return (options, 0)


# region User Combos
def _selected(self, selected):
    for i, o in enumerate(self.options):
        if o == selected:
            return (selected, i + 1, self.options)


class MpiCombo1:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 1 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_1.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo2:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 2 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_2.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo3:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 3 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_3.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo4:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 4 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_4.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo5:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 5 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_5.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo6:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 6 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_6.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo7:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 7 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_7.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo8:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 8 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_8.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo9:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 9 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_9.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


class MpiCombo10:
    CATEGORY = "MpiNodes/Combos"
    DESCRIPTION = "Select from combo 10 options"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("selected_string", "selected_index", "options")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        cls.options = load_combo("combo_10.txt")
        return {
            "required": {
                "selected": (cls.options,),
            }
        }

    def select(self, selected):
        return _selected(self, selected)


# endregion
