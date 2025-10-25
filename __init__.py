from .prompt_gen.prompt_list import MpiPromptList
from .prompt_gen.prompt_list_selector import MpiPromptListSelector
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
from .img import (
    MpiBboxToMask,
    MpiScaledDimensions,
    MpiAspectRatio,
    MpiGetImageAtIndex,
    MpiMaskDebugInfo,
)
from .text_list_replace import MpiTextListReplace
from .text_list_join import MpiTextListJoin
from .normalize_value import MpiNormalizeValue
from .logic import (
    MpiBoolean,
    MpiSimpleBoolean,
    MpiBooleanCompare,
    MpiFloat,
    MpiInt,
    MpiString,
    MpiText,
)
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
from .json import MpiJsonLoad, MpiJsonSave
from .if_else import (
    MpiIfElse,
    MpiIfElseInverted,
    MpiIfElseProbability,
    MpiIfElseProbabilityInverted,
)
from .checkpoint import MpiFromCheckpoint, MpiToChekPoint
from .multi_text import (
    MpiMultiTextX2,
    MpiMultiTextX3,
    MpiMultiTextX4,
    MpiMultiTextX5,
)

# TEST NODES
# from .dynamic_combo import MpiDynamicCombo
# from .test_dummy import MpiDummyToggleNode
# from .subgraph_test import MpiSubGraphTest


NODE_CLASS_MAPPINGS = {
    "MpiBatchTextReplace": MpiBatchTextReplace,
    "MpiBboxToMask": MpiBboxToMask,
    "MpiAddImageToList": MpiAddImageToList,
    "MpiScaledDimensions": MpiScaledDimensions,
    "MpiTextListReplace": MpiTextListReplace,
    "MpiTextListJoin": MpiTextListJoin,
    "MpiPromptList": MpiPromptList,
    "MpiPromptListSelector": MpiPromptListSelector,
    "MpiPromptProcessor": MpiPromptProcessor,
    "MpiNormalizeValue": MpiNormalizeValue,
    "MpiPromptChainSelector": MpiPromptChainSelector,
    "MpiPromptListJoin": MpiPromptListJoin,
    "MpiBoolean": MpiBoolean,
    "MpiSimpleBoolean": MpiSimpleBoolean,
    "MpiBooleanCompare": MpiBooleanCompare,
    "MpiFloat": MpiFloat,
    "MpiInt": MpiInt,
    "MpiString": MpiString,
    "MpiText": MpiText,
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
    "MpiGetImageAtIndex": MpiGetImageAtIndex,
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
    "MpiIfElseInverted": MpiIfElseInverted,
    "MpiIfElseProbability": MpiIfElseProbability,
    "MpiIfElseProbabilityInverted": MpiIfElseProbabilityInverted,
    "MpiFromCheckpoint": MpiFromCheckpoint,
    "MpiToChekPoint": MpiToChekPoint,
    "MpiMultiTextX2": MpiMultiTextX2,
    "MpiMultiTextX3": MpiMultiTextX3,
    "MpiMultiTextX4": MpiMultiTextX4,
    "MpiMultiTextX5": MpiMultiTextX5,
    "MpiMaskDebugInfo": MpiMaskDebugInfo,
    # TESTS
    # "MpiDummyToggleNode": MpiDummyToggleNode,
    # "MpiDynamicCombo": MpiDynamicCombo,
    # "MpiSubGraphTest": MpiSubGraphTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MpiBatchTextReplace": "Mpi Batch Text Replace",
    "MpiBboxToMask": "Mpi Bbox To Mask",
    "MpiAddImageToList": "Mpi Add Image to List",
    "MpiScaledDimensions": "Mpi Scaled Dimensions",
    "MpiTextListReplace": "Mpi Text List Replace",
    "MpiTextListJoin": "Mpi Text List Join",
    "MpiPromptList": "Mpi Prompt List",
    "MpiPromptListSelector": "Mpi Prompt List Selector",
    "MpiPromptProcessor": "Mpi Prompt Processor",
    "MpiNormalizeValue": "Mpi Normalize Value",
    "MpiPromptChainSelector": "Mpi Prompt Chain Selector",
    "MpiPromptListJoin": "Mpi Prompt List Join",
    "MpiBoolean": "Mpi Boolean",
    "MpiSimpleBoolean": "Mpi Simple Bool",
    "MpiBooleanCompare": "Mpi Boolean Compare",
    "MpiFloat": "Mpi Float",
    "MpiInt": "Mpi Int",
    "MpiString": "Mpi String",
    "MpiText": "Mpi Text",
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
    "MpiGetImageAtIndex": "Mpi Get Image At Index",
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
    "MpiIfElseProbability": "Mpi If Else Probability",
    "MpiIfElseProbabilityInverted": "Mpi If Else Probability Inverted",
    "MpiIfElseInverted": "Mpi If Else Inverted",
    "MpiFromCheckpoint": "Mpi From Checkpoint",
    "MpiToChekPoint": "Mpi To ChekPoint",
    "MpiMultiTextX2": "Mpi Multi Text x2",
    "MpiMultiTextX3": "Mpi Multi Text x3",
    "MpiMultiTextX4": "Mpi Multi Text x4",
    "MpiMultiTextX5": "Mpi Multi Text x5",
}

WEB_DIRECTORY = "./web"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
