# ComfyUi-MpiNodes

A ComfyUI custom node pack by [Mad Pony Interactive](https://github.com/MadPonyInteractive) with ~60 utilities for logic, math, prompt generation, image operations, model management, and workflow automation.

[![Registry](https://img.shields.io/badge/Comfy%20Registry-mad--pony--interactive-blue)](https://registry.comfy.org/publishers/mad-pony-interactive/nodes/ComfyUi-MpiNodes)


---

## Nodes

### Prompt Generation
Build, process, randomize, and route prompts with fine-grained control.

| Node | Description |
|---|---|
| **MpiPromptList** | Create a prompt list with options, probability, and blocking rules. Supports prefix/suffix, amount selection, specific item picking, probabilistic shuffling, and chaining. |
| **MpiPromptProcessor** | Process prompt lists into positive and negative prompts by replacing `_title_` tokens with selected options. Handles multipliers (e.g. `_title_x2_`) and cleanup. |
| **MpiPromptListSelector** | Probabilistically select between two prompt lists based on seed and probability. |
| **MpiPromptChainSelector** | Select one list from a prompt chain using a seed. |
| **MpiPromptListJoin** | Join two prompt lists or chains. |
| **MpiPromptBridge** | Pass-through utility that forwards a prompt list and seed values. |
| **MpiPromptOverride** | Override prompt list options with custom text and optional seed override. |
| **MpiHasOptions** | Check if selected options contain any of the specified options. Returns a boolean. |
| **MpiDisplayPrompt** | Display prompt list contents for debugging — outputs titles, blocklists, and processed prompts as strings. |
| **MpiRandPromptGen** | Generate random prompts from preset lists with customizable sections (shot type, hair, clothes, pose, expression, location, lighting). Supports indoor/outdoor probability, pronoun selection, and bright/dark lighting. |
| **MpiRandPromptGenOrder** | Set custom section order and transitions for random prompt generation. |
| **MpiRandPromptGenOverrideSection** | Override specific section options and seed for random prompt generation. |
| **MpiRandPromptGenOverrideList** | Override preset list options for random prompt generation. |
| **MpiRandPromptGenSave** | Save a random prompt preset to disk. |

---

### Logic & Control Flow
Primitive operations for comparisons, type conversions, rounding, and boolean logic.

| Node | Description |
|---|---|
| **MpiCompare** | General logic operator comparing two values with `==`, `!=`, `>`, `<`, `>=`, `<=`. |
| **MpiBooleanCompare** | Compare two booleans with modes: equal, not_equal, both_false, both_true, one_is_true, not_both_true, etc. |
| **MpiConvert** | Convert any value to STRING, INT, FLOAT, and BOOLEAN outputs simultaneously. |
| **MpiRoundToMultiple** | Round a single integer to the nearest multiple (up or down). |
| **MpiRoundToMultipleRes** | Round width and height to the nearest multiple simultaneously. |
| **MpiBoolean** | Pass through a boolean → outputs boolean, int, and float. |
| **MpiSimpleBoolean** | Pass through a boolean value only. |
| **MpiFloat / MpiInt / MpiString / MpiText** | Pass-through nodes for float, int, string (single-line), and text (multiline) values. |
| **MpiLogger** | Log any input value to the console with a prefix. |

---

### If / Else Routing
Conditional and probabilistic routing of values and inputs.

| Node | Description |
|---|---|
| **MpiIfElse** | Output true or false value based on a boolean condition (lazy evaluation). |
| **MpiIfElseInverted** | Route input to output A or output B based on a boolean condition. |
| **MpiIfElseProbability** | Probabilistically choose between true and false using seed and probability. |
| **MpiIfElseProbabilityInverted** | Probabilistically route input to true or false output. |

---

### Switches
Route any type of input to a selected output by index.

| Node | Description |
|---|---|
| **MpiAnySwitch** | Select one of up to 5 inputs of any type based on selection index. |
| **MpiLoraSwitch** | Select one of up to 5 LoRA files by index. |
| **MpiAnyInvSwitch** | Route an input of any type to one of up to 5 outputs (inverted switch). |
| **MpiStringInvSwitch** | Route a string input to one of up to 5 outputs. |

---

### Combos
Dropdown selectors for reusable, named option lists.

| Node | Description |
|---|---|
| **MpiComboMaker** | Create, edit, or delete custom combo options stored in `web/user/combos.json`. |
| **MpiCombo** | Select from custom combo options (created via ComboMaker) — returns value and index. |
| **MpiCombo1 – MpiCombo10** | Ten preset combo nodes reading from `combo_1.txt` through `combo_10.txt` in `web/user/combos/`. Return selected string, index, and all options. |

---

### Image Operations
Dimension math, aspect ratio, bounding box conversion, and grid tiling.

| Node | Description |
|---|---|
| **MpiScaledDimensions** | Scale image dimensions proportionally to a target size (use_max or use_min side). Returns width, height, and is_portrait boolean. |
| **MpiAspectRatio** | Calculate aspect ratio from width/height (returns 1:1, 4:3, 3:2, 16:9, 9:16, 2:3, 3:4). |
| **MpiGetImageAtIndex** | Return the image at a specified index from a batch. Supports negative indexing (-1 = last). |
| **MpiBboxToMask** | Convert bounding boxes (xyxy or xywh format) to mask tensors. |
| **MpiGridDimensions** | Calculate grid dimensions and corrected source size for perfect tiling — avoids repeated tiles when fed to UltimateSDUpscale. Has auto mode. |
| **MpiMaskDebugInfo** | Print mask shape, dtype, and device info to the console for debugging. |
| **MpiAddImageToList** | Append an image to a list of images. |

---

### Math

| Node | Description |
|---|---|
| **MpiMath** | Evaluate a Python expression on inputs `a`, `b`, `c` (b and c optional). All `math` module functions are in scope. |
| **MpiExpoFloat** | Apply a power curve to a 0–1 float: `value ^ exponent`, clamped to [0, 1]. |
| **MpiNormalizeValue** | Normalize a value from an input range to an output range with optional transforms (linear, inverse, log, exp) and clamping. |

---

### LoRA & Checkpoint

| Node | Description |
|---|---|
| **MpiLoraModel** | Apply a LoRA to a model only, with strength control. |
| **MpiLoraModelClip** | Apply a LoRA to both model and clip with independent strength controls. |
| **MpiDoubleLora** | Apply the same LoRA to two models and clips simultaneously with strength controls. |
| **MpiFromCheckpoint** | Extract model, clip, and VAE from a checkpoint. |
| **MpiToCheckpoint** | Combine model, clip, and VAE into a checkpoint. |

---

### Conditioning

| Node | Description |
|---|---|
| **MpiMultiTextX2 / X3 / X4 / X5** | Combine 2, 3, 4, or 5 text inputs into one. |

---

### Text & String Operations

| Node | Description |
|---|---|
| **MpiBatchTextReplace** | Batch find and replace text in all `.txt` files within a folder. Handles UTF-8 and cp1252 encoding. |
| **MpiTextListReplace** | Find and replace text in each item of a text list. |
| **MpiTextListJoin** | Prepend or append text to each item in a text list. |

---

### Animation Timing (Wan)

| Node | Description |
|---|---|
| **MpiWanFrames** | Calculate duration from frame count and FPS. Returns frames, seconds, and fps. |
| **MpiWanSeconds** | Calculate frame count from duration (seconds) and FPS. Returns frames, seconds, and fps. |

---

### JSON

| Node | Description |
|---|---|
| **MpiJsonLoad** | Load a JSON file from a full path and output as a dictionary. |
| **MpiJsonSave** | Save a dictionary to a JSON file at a specified path. |

---

### Utilities

| Node | Description |
|---|---|
| **MpiClearVram** | Offload all models from VRAM to RAM, clear GPU cache, and run garbage collection. Passes through a value to chain in workflows. |

---

## License

[MIT](https://choosealicense.com/licenses/mit/)
