import os, pprint, copy
from typing import TypedDict, cast
from collections import Counter
from server import PromptServer  # type: ignore
import folder_paths as comfy_paths  # type: ignore

from ..help_funcs import (
    load_json_dict,
    load_json_dict_abs,
    list_text_files,
    read_lines,
    seeded_shuffle,
    get_random_bool,
    copy_folder,
    write_lines_to_file,
    save_json_dict_abs,
)


class Lists(TypedDict):
    bikini: list[str]
    expressions: list[str]
    hair_styles: list[str]
    indoor_bright_lighting: list[str]
    indoor_clothing_set: list[str]
    indoor_dark_lighting: list[str]
    indoor_locations: list[str]
    lingerie: list[str]
    outdoor_bright_lighting: list[str]
    outdoor_clothing_set: list[str]
    outdoor_dark_lighting: list[str]
    outdoor_locations: list[str]
    poses: list[str]
    shot_types: list[str]


def _get_lists():
    """Return ["indoor_locations", "outdoor_locations", etc]"""
    return list(Lists.__annotations__.keys())


_sections: list = [
    "shot_type",
    "pre_positive",
    "hair",
    "clothes",
    "pose",
    "expression",
    "location",
    "light",
    "post_positive",
]
_clean_sections: list = [
    item for item in _sections if item not in ["pre_positive", "post_positive"]
]


class RPGPreset:
    def __init__(self, root_folder: str, overrides: dict = {}) -> None:
        self.file_list: list[str] = []

        preset_path = os.path.join(
            comfy_paths.get_folder_paths("custom_nodes")[0],
            "ComfyUi-MpiNodes/user/prompt_gen_preset/",
        )
        self.root_folder: str = root_folder or preset_path

        self.file_list = list_text_files(self.root_folder)
        if self.root_folder and not self.file_list:
            raise ValueError(
                "MpiRandPromptGen: No files found in current preset folder!"
            )

        # remove read_me.txt and .txt extensions
        self.file_list = [
            item.removesuffix(".txt")
            for item in self.file_list
            if item != "read_me.txt"
        ]

        # check if lists are a match
        assert set(self.file_list) == set(
            Lists.__annotations__
        ), "MpiRandPromptGen: Preset is not valid, lists don't match!"

        # Make lists
        self.lists: Lists = cast(
            Lists, {key: [] for key in Lists.__annotations__}
        )
        for file in self.file_list:
            if file in self.lists:
                self.lists[file] = read_lines(
                    os.path.join(self.root_folder, file + ".txt")
                )

        self.ovr_list: list[dict] = list(copy.deepcopy(overrides).values())
        self.overriden_sections = self.filter_by_type("section")
        self.overriden_lists = self.filter_by_type("list")
        self.overriden_order = self.filter_by_type("order")

        self.order = self.settings_file.get("order", {})
        self.sections = (
            self.overriden_order.get("sections", [])
            or self.order.get("sections", [])
            or _sections
        )
        self.transitions = self.overriden_order.get(
            "transitions", []
        ) or self.order.get("transitions", [])

    # def __getitem__(self, key):
    #     return self.lists[key]

    @property
    def settings_file(self):
        """get settings from settings.json"""
        return load_json_dict_abs(
            os.path.join(self.root_folder, "settings.json")
        )

    def filter_by_type(self, _type: str) -> dict:
        out: dict = {}
        for d in self.ovr_list:
            if d.get("type", "") == _type:
                d.pop("type")
                if _type == "order":
                    out = d
                else:
                    out[d.pop(_type, "")] = d
        return out


class MpiRandPromptGenOrder:
    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("MPI_RANDPROMPT_OVR",)
    RETURN_NAMES = ("overrides",)
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_1": (_sections, {"default": _sections[0]}),
                "transition_1": ("STRING", {"default": ". "}),
                "section_2": (_sections, {"default": _sections[1]}),
                "transition_2": ("STRING", {"default": ""}),
                "section_3": (_sections, {"default": _sections[2]}),
                "transition_3": ("STRING", {"default": ". "}),
                "section_4": (_sections, {"default": _sections[3]}),
                "transition_4": ("STRING", {"default": ". "}),
                "section_5": (_sections, {"default": _sections[4]}),
                "transition_5": ("STRING", {"default": ". "}),
                "section_6": (_sections, {"default": _sections[5]}),
                "transition_6": ("STRING", {"default": ". "}),
                "section_7": (_sections, {"default": _sections[6]}),
                "transition_7": ("STRING", {"default": ". "}),
                "section_8": (_sections, {"default": _sections[7]}),
                "transition_8": ("STRING", {"default": ". "}),
                "section_9": (_sections, {"default": _sections[8]}),
                "transition_9": ("STRING", {"default": ""}),
            },
            "optional": {"overrides": ("MPI_RANDPROMPT_OVR",)},
        }

    def doit(self, overrides={}, **kw):
        _id = id(self)
        _overrides = copy.deepcopy(overrides)
        _overrides[_id] = {
            "type": "order",
            "sections": [v for k, v in kw.items() if "section_" in k],
            "transitions": [v for k, v in kw.items() if "transition_" in k],
        }
        return (_overrides,)


class MpiRandPromptGenOverrideList:
    """Used to override preset lists"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": (_get_lists(), {"tooltip": "Preset lists"}),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "When set, overrides the list and affects saved preset!",
                    },
                ),
            },
            "optional": {"overrides": ("MPI_RANDPROMPT_OVR",)},
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("MPI_RANDPROMPT_OVR",)
    RETURN_NAMES = ("overrides",)
    FUNCTION = "doit"

    def doit(
        self,
        list: str,
        options: str,
        overrides: dict = {},
    ):
        _id = id(self)
        _overrides = copy.deepcopy(overrides)
        _overrides[_id] = {
            "type": "list",
            "list": list,
            "options": options.splitlines(),
        }
        return (_overrides,)


class MpiRandPromptGenOverrideSection:
    """Used to override current sections and their seeds.
    Does not affect the preset in any manner"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section": (_clean_sections, {}),
                "options": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "When set, overrides the list",
                    },
                ),
                "seed_override": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": """If not -1 it overrides the current seed changing 
                                the resulting random selection for the selected section. 
                                Does not affect the global seed!""",
                    },
                ),
            },
            "optional": {"overrides": ("MPI_RANDPROMPT_OVR", {})},
        }

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("MPI_RANDPROMPT_OVR",)
    RETURN_NAMES = ("overrides",)
    FUNCTION = "doit"

    def doit(
        self,
        section: str,
        options: str,
        seed_override: int,
        overrides: dict = {},
    ):
        _id = id(self)
        _overrides = copy.deepcopy(overrides)
        _overrides[_id] = {
            "type": "section",
            "section": section,
            "options": options.splitlines(),
            "seed": seed_override,
        }
        return (_overrides,)


class MpiRandPromptGen:
    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "MPI_PROMPT_PRESET")
    RETURN_NAMES = ("positive", "negative", "sections", "save_preset")
    FUNCTION = "ret_func"
    # OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        cls.defaults = load_json_dict("user/prompt_gen_combos.json")
        return {
            "required": {
                "preset_folder": (
                    "STRING",
                    {"tooltip": "path/to/preset_folder"},
                ),
                "pre_positive": (
                    "STRING",
                    {"multiline": True},
                ),
                "indoors_outdoors": (
                    "INT",
                    {
                        "default": 50,
                        "min": 0,
                        "max": 100,
                        "tooltip": "0 = indoors\n100 = outdoors\n50 = 50% chance of either",
                    },
                ),
                "pronoun": (["random"] + cls.defaults.get("pronouns", []),),
                "shot_type": (
                    ["random", "none"] + cls.defaults.get("shot_types", []),
                ),
                "hair_styles": (
                    ["random", "none"] + cls.defaults.get("hair_styles", []),
                ),
                "clothes": (["random", "nude", "set", "bikini", "lingerie"],),
                "pose": (["random", "none"] + cls.defaults.get("poses", []),),
                "expression": (
                    ["random", "none"] + cls.defaults.get("expressions", []),
                ),
                "light_type": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "bright",
                        "label_off": "dark",
                    },
                ),
                "post_positive": (
                    "STRING",
                    {"multiline": True},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
            },
            "optional": {
                "overrides": ("MPI_RANDPROMPT_OVR",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, preset_folder, **kw):
        return preset_folder

    def ret_func(self, preset_folder: str, overrides: dict = {}, **kw):
        Pg = RPGPreset(preset_folder, overrides)

        seed: int = kw.get("seed", 0)

        in_out_p: int = kw.get("indoors_outdoors", 0)
        outdoors: bool = get_random_bool(in_out_p, seed) if in_out_p else False
        bright_light: bool = kw.get("light_type", True)

        pronoun: str = kw.get("pronoun", "random")
        if pronoun == "random":
            pronoun = seeded_shuffle(self.defaults.get("pronouns", []), seed)[
                0
            ]

        def get_seed(section):
            s = Pg.overriden_sections.get(section, {}).get("seed", -1)
            return s if s != -1 else seed

        def get_list(_list):
            """return list or overriden list"""
            return (
                Pg.overriden_lists.get(_list, {}).get("options", [])
                or Pg.lists[_list]
            )

        def shuffle(section, _list):
            return seeded_shuffle(
                get_list(_list),
                get_seed(section),
            )[0]

        def override_section(section, selection):
            lst = Pg.overriden_sections.get(section, {}).get("options", [])
            if not lst:
                return selection
            return seeded_shuffle(
                Pg.overriden_sections.get(section, {}).get("options", []),
                get_seed(section),
            )[0]

        # Lists Shuffle
        hair_style: str = kw.get("hair_styles", "random")
        if hair_style == "random":
            hair_style = shuffle("hair", "hair_styles")
        elif hair_style == "none":
            hair_style = ""

        pose: str = kw.get("pose", "random")
        if pose == "random":
            pose = shuffle("pose", "poses")
        elif pose == "none":
            pose = ""

        expression: str = kw.get("expression", "random")
        if expression == "random":
            expression = shuffle("expression", "expressions")
        elif expression == "none":
            expression = ""

        shot_type: str = kw.get("shot_type", "random")
        if shot_type == "random":
            shot_type = shuffle("shot_type", "shot_types")
        elif shot_type == "none":
            shot_type = ""

        clothes: str = kw.get("clothes", "random")
        if clothes == "random":
            clothes = seeded_shuffle(
                ["set", "bikini", "lingerie", "nude"], get_seed("clothes")
            )[0]
        clothes_lst: str = ""
        if clothes == "set":
            if outdoors:
                clothes_lst = "outdoor_clothing_set"
            else:
                clothes_lst = "indoor_clothing_set"
        elif clothes == "bikini":
            clothes_lst = "bikini"
        elif clothes == "lingerie":
            clothes_lst = "lingerie"
        if clothes != "nude":
            clothes = shuffle("clothes", clothes_lst)

        location_lst: str = ""
        if outdoors:
            location_lst = "outdoor_locations"
        else:
            location_lst = "indoor_locations"
        location = shuffle("location", location_lst)

        light_lst: str = ""
        if outdoors:
            if bright_light:
                light_lst = "outdoor_bright_lighting"
            else:
                light_lst = "outdoor_dark_lighting"
        else:
            if bright_light:
                light_lst = "indoor_bright_lighting"
            else:
                light_lst = "indoor_dark_lighting"
        light = shuffle("light", light_lst)

        # Section adjustments and overrides
        light = override_section("light", light)

        # Fix pronoun
        p_is = "are" if pronoun == "they" else "is"
        p_has = "have" if pronoun == "they" else "has"

        if location:
            if outdoors:
                location = f"{pronoun} {p_is} {location}"
            else:
                location = f"{pronoun} {p_is} in a {location}"
        location = override_section("location", location)

        if hair_style:
            hair_style = f"{pronoun} {p_has} {hair_style}"
        hair_style = override_section("hair", hair_style)

        if clothes:
            if clothes == "nude":
                clothes = f"{pronoun} {p_is} nude"
            else:
                clothes = f"{pronoun} {p_is} wearing a {clothes}"
        clothes = override_section("clothes", clothes)

        if expression:
            expression = f"{pronoun} {p_has} a {expression} expression"
        expression = override_section("expression", expression)

        if pose:
            pose = f"{pronoun} {p_is} {pose}"
        pose = override_section("pose", pose)

        shot_type = override_section("shot_type", shot_type)

        sections: dict = {
            "location": location,
            "light": light,
            "clothes": clothes,
            "pose": pose,
            "expression": expression,
            "hair": hair_style,
            "shot_type": shot_type,
        }
        # Build Positive
        positive: str = ""
        output_sections: list = []
        for sec, trs in zip(Pg.sections, Pg.transitions):
            section: str
            if sec in ["pre_positive", "post_positive"]:
                section = kw.get(sec, "")
            else:
                section = sections[sec].capitalize()
            if section:
                sec_content = f"{section}{trs}"
                positive += sec_content
                output_sections.append(f"{sec.upper()}\n{sec_content}")

        return (positive, "neg", output_sections, Pg)


class MpiRandPromptGenLoadPreset:
    """Saves a path to a temp file that can be accessed by other nodes"""

    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_folder": (
                    "STRING",
                    {
                        "default": "path/to/preset_folder",
                        "tooltip": "To load the preset into an existing RandPromptGen you need to run this node and then press 'R' to refresh the RandPromptGen preset list!",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    @classmethod
    def IS_CHANGED(cls, preset_folder, **kw):
        return preset_folder

    def doit(self, preset_folder: str, unique_id, prompt):
        # from .... import folder_paths as comfy_paths

        # print(unique_id)

        preset_loaders = [
            v
            for _id, v, in prompt.items()
            if v.get("class_type", "") == "MpiRandPromptGenLoadPreset"
            # and _id != unique_id
        ]
        # if preset_loaders:
        #     raise ValueError("Only 1 RandromptGen preset supported!")

        preset_folders = [
            d.get("inputs", {}).get("preset_folder", "")
            for d in preset_loaders
        ]

        temp = comfy_paths.get_temp_directory()
        temp_file = os.path.join(temp, "MpiPresets.json")
        # presets = load_json_dict_abs(temp_file)

        presets: dict = {}
        for folder in preset_folders:
            if not os.path.exists(folder):
                continue
            sett_file = os.path.join(folder, "settings.json")
            if not os.path.exists(sett_file):
                continue
            presets[folder] = load_json_dict_abs(sett_file)

        save_json_dict_abs(presets, temp_file)
        # pprint.pp(presets)

        return ()


class MpiRandPromptGenSave:
    CATEGORY = "MpiNodes/PromptGen"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("destination",)
    OUTPUT_NODE = True
    FUNCTION = "doit"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (
                    "MPI_PROMPT_PRESET",
                    {"forceInput": True},
                ),
                "preset_name": (
                    "STRING",
                    {
                        "default": "my_new_preset",
                    },
                ),
                "destination_folder": (
                    "STRING",
                    {
                        "default": "path/to/save/preset_folder",
                    },
                ),
            },
        }

    def doit(
        self, preset: RPGPreset, preset_name: str, destination_folder: str
    ):
        # Copy folder
        dest_folder = os.path.join(destination_folder, preset_name)
        # copy folder contents if the destination folder does not exist
        if not os.path.isdir(dest_folder):
            copy_folder(preset.root_folder, dest_folder)
        # Update file lists
        for filename, value in preset.overriden_lists.items():
            options = value.get("options", [])
            filename += ".txt"
            file_path = os.path.join(dest_folder, filename)
            write_lines_to_file(file_path, options)
        # Remove read_me_.txt
        read_me = os.path.join(dest_folder, "read_me.txt")
        if os.path.isfile(read_me):
            os.remove(read_me)
        # Update settings file
        settings = preset.settings_file
        settings["order"] = {
            "sections": preset.sections,
            "transitions": preset.transitions,
        }
        save_json_dict_abs(
            settings, os.path.join(dest_folder, "settings.json")
        )
        return (dest_folder,)
