from .prompt_gen.prompt_list import MpiPromptList
from .prompt_gen.prompt_list_selector import (
    MpiPromptListSelector,
    MpiPromptListSelectorBool,
    MpiPromptListSelectorBoolInverted,
)
from .prompt_gen.prompt_processor import MpiPromptProcessor
from .prompt_gen.prompt_chain_selector import MpiPromptChainSelector
from .prompt_gen.prompt_list_join import MpiPromptListJoin
from .prompt_gen.has_options import MpiHasOptions
from .prompt_gen.prompt_bridge import MpiPromptBridge
from .prompt_gen.prompt_override import (
    MpiPromptOverride,
    MpiPromptOverrideSeed,
)
from .prompt_gen.display_prompts import MpiDisplayPrompt
from .prompt_gen.rand_prompt_gen import (
    MpiRandPromptGenOrder,
    MpiRandPromptGenOverrideSection,
    MpiRandPromptGenOverrideList,
    MpiRandPromptGen,
    MpiRandPromptGenSave,
)
from .batch_text_replace import MpiBatchTextReplace
from .add_image_to_list import MpiAddImageToList
from .text_list_replace import MpiTextListReplace
from .text_list_join import MpiTextListJoin
from .normalize_value import MpiNormalizeValue
from .boolean import MpiBoolean, MpiBooleanCompare
from .switches import (
    MpiStringInvSwitch,
    MpiAnyInvSwitch,
    MpiLoraSwitch,
    MpiAnySwitch,
)
from .combos import (
    MpiCombo1,
    MpiCombo2,
    MpiCombo3,
    MpiCombo4,
    MpiCombo5,
    MpiCombo6,
    MpiCombo7,
    MpiCombo8,
    MpiCombo9,
    MpiCombo10,
    MpiCombo,
    MpiComboMaker,
)
from .loras import MpiDoubleLora, MpiLoraModel
from .get_ratio import MpiAspectRatio
from .json import MpiJsonLoad, MpiJsonSave
from .if_else import MpiIfElse
from .checkpoint import MpiFromCheckpoint, MpiToChekPoint

# TEST NODES
# from .dynamic_combo import MpiDynamicCombo
# from .test_dummy import MpiDummyToggleNode
# from .subgraph_test import MpiSubGraphTest


NODE_CLASS_MAPPINGS = {
    "MpiBatchTextReplace": MpiBatchTextReplace,
    "MpiAddImageToList": MpiAddImageToList,
    "MpiTextListReplace": MpiTextListReplace,
    "MpiTextListJoin": MpiTextListJoin,
    "MpiPromptList": MpiPromptList,
    "MpiPromptListSelector": MpiPromptListSelector,
    "MpiPromptListSelectorBool": MpiPromptListSelectorBool,
    "MpiPromptListSelectorBoolInverted": MpiPromptListSelectorBoolInverted,
    "MpiPromptProcessor": MpiPromptProcessor,
    "MpiNormalizeValue": MpiNormalizeValue,
    "MpiPromptChainSelector": MpiPromptChainSelector,
    "MpiPromptListJoin": MpiPromptListJoin,
    "MpiBoolean": MpiBoolean,
    "MpiBooleanCompare": MpiBooleanCompare,
    "MpiDisplayPrompt": MpiDisplayPrompt,
    "MpiHasOptions": MpiHasOptions,
    "MpiPromptBridge": MpiPromptBridge,
    "MpiPromptBridge": MpiPromptBridge,
    "MpiStringInvSwitch": MpiStringInvSwitch,
    "MpiAnyInvSwitch": MpiAnyInvSwitch,
    "MpiLoraSwitch": MpiLoraSwitch,
    "MpiAnySwitch": MpiAnySwitch,
    "MpiCombo1": MpiCombo1,
    "MpiCombo2": MpiCombo2,
    "MpiCombo3": MpiCombo3,
    "MpiCombo4": MpiCombo4,
    "MpiCombo5": MpiCombo5,
    "MpiCombo6": MpiCombo6,
    "MpiCombo7": MpiCombo7,
    "MpiCombo8": MpiCombo8,
    "MpiCombo9": MpiCombo9,
    "MpiCombo10": MpiCombo10,
    "MpiCombo": MpiCombo,
    "MpiComboMaker": MpiComboMaker,
    "MpiDoubleLora": MpiDoubleLora,
    "MpiLoraModel": MpiLoraModel,
    "MpiAspectRatio": MpiAspectRatio,
    "MpiPromptOverride": MpiPromptOverride,
    "MpiPromptOverrideSeed": MpiPromptOverrideSeed,
    "MpiJsonLoad": MpiJsonLoad,
    "MpiJsonSave": MpiJsonSave,
    "MpiRandPromptGenOrder": MpiRandPromptGenOrder,
    "MpiRandPromptGenOverrideSection": MpiRandPromptGenOverrideSection,
    "MpiRandPromptGenOverrideList": MpiRandPromptGenOverrideList,
    "MpiRandPromptGenSave": MpiRandPromptGenSave,
    "MpiRandPromptGen": MpiRandPromptGen,
    "MpiIfElse": MpiIfElse,
    "MpiFromCheckpoint": MpiFromCheckpoint,
    "MpiToChekPoint": MpiToChekPoint,
    # TESTS
    # "MpiDummyToggleNode": MpiDummyToggleNode,
    # "MpiDynamicCombo": MpiDynamicCombo,
    # "MpiSubGraphTest": MpiSubGraphTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiBatchTextReplace": "Mpi Batch Text Replace",
    "MpiAddImageToList": "Mpi Add Image to List",
    "MpiTextListReplace": "Mpi Text List Replace",
    "MpiTextListJoin": "Mpi Text List Join",
    "MpiPromptList": "Mpi Prompt List",
    "MpiPromptListSelector": "Mpi Prompt List Selector",
    "MpiPromptListSelectorBool": "Mpi Prompt List Selector Bool",
    "MpiPromptListSelectorBoolInverted": "Mpi Prompt List Selector Bool Inverted",
    "MpiPromptProcessor": "Mpi Prompt Processor",
    "MpiNormalizeValue": "Mpi Normalize Value",
    "MpiPromptChainSelector": "Mpi Prompt Chain Selector",
    "MpiPromptListJoin": "Mpi Prompt List Join",
    "MpiBoolean": "Mpi Boolean",
    "MpiBooleanCompare": "Mpi Boolean Compare",
    "MpiDisplayPrompt": "Mpi Prompt Display",
    "MpiHasOptions": "Mpi Has Options",
    "MpiPromptBridge": "Mpi Prompt Bridge",
    "MpiLoraSwitch": "Mpi Lora Switch",
    "MpiAnySwitch": "Mpi Any Switch",
    "MpiStringInvSwitch": "Mpi String Inverted Switch",
    "MpiAnyInvSwitch": "Mpi Any Inverted Switch",
    "MpiChoiceSelector": "Mpi Choice Selector",
    "MpiPresetSelector": "Mpi Preset Selector",
    "MpiCombo1": "Mpi Combo1",
    "MpiCombo2": "Mpi Combo2",
    "MpiCombo3": "Mpi Combo3",
    "MpiCombo4": "Mpi Combo4",
    "MpiCombo5": "Mpi Combo5",
    "MpiCombo6": "Mpi Combo6",
    "MpiCombo7": "Mpi Combo7",
    "MpiCombo8": "Mpi Combo8",
    "MpiCombo9": "Mpi Combo9",
    "MpiCombo10": "Mpi Combo10",
    "MpiCombo": "Mpi Combo [BETA]",
    "MpiComboMaker": "Mpi Combo Maker [BETA]",
    "MpiDoubleLora": "Mpi Double Lora Loader",
    "MpiLoraModel": "Mpi Lora Model",
    "MpiAspectRatio": "Mpi Get Aspect Ratio",
    "MpiPromptOverride": "Mpi Prompt Override",
    "MpiPromptOverrideSeed": "Mpi Prompt Override Seed",
    "MpiJsonLoad": "Mpi Json Load",
    "MpiJsonSave": "Mpi Json Save",
    "MpiRandPromptGen": "Mpi RandPromptGen",
    "MpiRandPromptGenSave": "Mpi RandPromptGen Save",
    "MpiRandPromptGenOrder": "Mpi RandPromptGen Order",
    "MpiRandPromptGenOverrideSection": "Mpi RandPromptGen Override Section",
    "MpiRandPromptGenOverrideList": "Mpi RandPromptGen Override List",
    "MpiIfElse": "Mpi If Else",
    "MpiFromCheckpoint": "Mpi From Checkpoint",
    "MpiToChekPoint": "Mpi To ChekPoint",
}

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
