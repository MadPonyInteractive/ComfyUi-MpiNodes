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
| **MpiCompare** | General logic operator comparing two values with `==`, `!=`, `>`, `<`, `>=`, `<=`. `b` can be wired or typed into the `b_value` widget. |
| **MpiBooleanCompare** | Compare two booleans with modes: equal, not_equal, both_false, both_true, one_is_true, not_both_true, etc. |
| **MpiBooleanInvert** | Invert a boolean — true in, false out. A pass-through NOT gate; wire-only input, no widget. |
| **MpiConvert** | Convert any value to STRING, INT, FLOAT, and BOOLEAN outputs simultaneously. |
| **MpiRoundToMultiple** | Round a single integer to the nearest multiple (up or down). |
| **MpiRoundToMultipleRes** | Round width and height to the nearest multiple simultaneously. |
| **MpiBoolean** | Pass through a boolean → outputs boolean, int, and float. |
| **MpiSimpleBoolean** | Pass through a boolean value only. |
| **MpiFloat / MpiInt / MpiString / MpiText** | Pass-through nodes for float, int, string (single-line), and text (multiline) values. |
| **MpiListCount** | Count entries in a list of any type. Outputs `count` (INT) and `has_items` (BOOLEAN, true if non-empty). |
| **MpiIsListEmpty** | Check whether a list of any type is empty. Outputs `is_empty` (BOOLEAN, true if zero items) and `count` (INT). Route `is_empty` into an if/else to skip nodes that would `IndexError` on an empty list. |
| **MpiBlockIfEmptyList** | Pass a list through, but block downstream execution if it is empty. Place before nodes that index into a list so an empty list halts the branch instead of throwing `IndexError`. Unlike MpiAnyBlocker, it receives the whole list, so a truly empty list still reaches it. |
| **MpiListRange** | Output a sub-range of any list using inclusive `start`/`end` indices. Negative indices count from the end. Outputs sliced list and its count. |
| **MpiTextContains** | Check whether any of several words or phrases (multiline box, separated by commas or new lines) appears in an input text as a whole word. Case-insensitive and plural-aware, so `cat` matches `a CAT sits` and `two cats` but not `catalogue`. Multi-word entries ignore spacing, so `bird in a tree` still matches across a line break. Irregular plurals (`child`/`children`) need their own entry. Outputs a BOOLEAN — use as a whitelist/blacklist check on a prompt. |
| **MpiReroute** | Pass any value through unchanged. Rename the node title for a labelled reroute. |
| **MpiConditioningReroute** | Pass `positive` and `negative` conditioning through unchanged — a labelled conditioning reroute. |
| **MpiAnyBlocker** | Pass any value through, but block downstream execution if it is empty (empty string/list/dict/None/zero-element tensor/empty audio). 0, 0.0 and False pass through. |
| **MpiBlocker** | Manual gate: pass the input through when the switch is on (continue), block the branch when off (block). The input is **lazy**, so blocking also skips everything feeding this node — not just what comes after it. |
| **MpiAnyChecker** | Pass any value through unchanged and output a `has_value` boolean — true if non-empty, false if empty. Same emptiness rules as MpiAnyBlocker. |
| **MpiSeedPassthrough** | Pass any value through and emit a `seed`. Forces the workflow to re-run every time (via `IS_CHANGED`) so seed-less workflows don't get stuck on cached outputs. Leave `any` unconnected to use as a pure seed source. |
| **MpiLogger** | End-of-chain logger with no output — prints when its input arrives. `mode="value"` logs the input value with a prefix; `mode="message"` logs only the message text, marking that the point was reached without dumping the input. |
| **MpiExecLogger** | Pass any input through to its output while logging a message to the console — wire inline to print workflow progress. |

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
| **MpiAnySwitch10** | Select one of up to 10 inputs of any type based on selection index. |
| **MpiLoraSwitch** | Select one of up to 5 LoRA files by index. |
| **MpiAnyInvSwitch** | Route an input of any type to one of up to 5 outputs (inverted switch). |
| **MpiPacker** | Bundle up to 5 values of *any* type (image + float + int + boolean + …) into one `MPI_PACK` wire. Unconnected slots stay empty. A pack is just a list, so a pack can go into another packer's slot — 5 packers into a packer carries 25 values on one wire. |
| **MpiUnpacker** | Unpack an `MPI_PACK` back into 5 outputs, in the same slot order they were packed. Slots that were empty block execution downstream, so nothing runs on a missing value. |
| **MpiPacker10** | Same as MpiPacker with 10 slots instead of 5. |
| **MpiUnpacker10** | Same as MpiUnpacker with 10 outputs instead of 5. Feeding a 10-slot pack into the 5-slot MpiUnpacker drops the tail and logs a warning. |
| **MpiComparePacks** | Compare two packs slot by slot. Outputs `equal` (true only if every slot matches) and `first_diff`, the 1-based slot of the first mismatch (0 when equal). Images and other tensors compare by content, and nested packs compare all the way down — this is the pack-safe alternative to MpiCompare, which raises on tensors. |
| **MpiStringInvSwitch** | Route a string input to one of up to 5 outputs. |

---

### Image Operations
Dimension math, aspect ratio, bounding box conversion, and grid tiling.

| Node | Description |
|---|---|
| **MpiScaledDimensions** | Scale image dimensions proportionally to a target size (use_max or use_min side). Returns width, height, is_portrait boolean, and the image resized to those dimensions (upscale_method selects the interpolation). |
| **MpiAspectRatio** | Calculate aspect ratio from width/height (returns 1:1, 4:3, 3:2, 16:9, 9:16, 2:3, 3:4). |
| **MpiGetImageAtIndex** | Return the image at a specified index from a batch. Supports negative indexing (-1 = last). |
| **MpiBboxToMask** | Convert bounding boxes (xyxy or xywh format) to mask tensors. |
| **MpiGridDimensions** | Calculate grid dimensions and corrected source size for perfect tiling — avoids repeated tiles when fed to UltimateSDUpscale. Has auto mode. |
| **MpiUpscaleModelScale** | Takes a `Load Upscale Model` node (or any UPSCALE_MODEL input) and reads its native scale (1x/2x/4x/8x) from the model's descriptor metadata. Outputs INT and FLOAT. fallback_scale used only if metadata is absent. |
| **MpiLoadImageFromPath** | Load an image from a file path with an in-graph preview. Outputs image, mask, width, height. A channel combo (alpha/red/green/blue) selects the mask source. Blocks downstream execution if the path is empty, unless block_if_empty is off (then outputs a blank 1x1 image so the graph continues). |
| **MpiCrop** | Crop an image to width/height at a chosen anchor (center/left/right/top/bottom). width/height of 0 keep that dimension full; the crop is floored to a multiple of divisible_by. |
| **MpiMaskSquareBbox** | Square bounding box around a mask, centered on the mask and clamped (shrunk if needed) to stay fully inside the image. Outputs a filled square MASK plus x, y, and side length. Optional padding around the tight box. |
| **MpiBox** | Build an `MPI_BOX` rectangle from width/height/x/y, where x/y are the **top-left** corner. One wire carries a region into any box-aware node instead of four loose INT sockets. Consumers clamp the box to the image, so out-of-bounds values are safe. |
| **MpiFromBox** | Unpack an `MPI_BOX` back into width, height, x, y integers — the escape hatch for feeding raw INTs to other nodes. |
| **MpiBoxCrop** | Crop an image to an `MPI_BOX` region. Outputs the cropped image plus the clamped box actually used. A box fully outside the image passes the image through unchanged. |
| **MpiBoxMask** | Build a mask the size of the image — black with a white rectangle at the `MPI_BOX` region. Outputs the mask plus the clamped box actually drawn. |
| **MpiInpaintHeal** | Heal an inpainted region against the real pixels around it — corrects colour drift and restores fine grain by matching a ring of untouched pixels hugging the mask. Unlike a colour-match on a rectangular crop, the ring never contains the removed object, so it cannot pull the fill toward it. Run after stitching. |
| **MpiMaskDebugInfo** | Print mask shape, dtype, and device info to the console for debugging. |
| **MpiAddImageToList** | Append an image to a list of images. |

---

### Math

| Node | Description |
|---|---|
| **MpiMath** | Evaluate a Python expression on inputs `a`, `b`, `c` (b and c optional). All `math` module functions are in scope. |
| **MpiExpoFloat** | Apply a power curve to a 0–1 float: `value ^ exponent`, clamped to [0, 1]. |
| **MpiClamp** | Clamp an int or float between `min_value` and `max_value`. With min 200 / max 1000: `400` stays `400`, `2000` becomes `1000`, `100` becomes `200`. Preserves int inputs. |
| **MpiNormalizeValue** | Normalize a value from an input range to an output range with optional transforms (linear, inverse, log, exp) and clamping. |

---

### LoRA & Checkpoint

| Node | Description |
|---|---|
| **MpiLoraModel** | Apply a LoRA to a model only, with strength control. |
| **MpiLoraModelClip** | Apply a LoRA to both model and clip with independent strength controls. |
| **MpiDoubleLora** | Apply the same LoRA to two models and clips simultaneously with strength controls. |
| **MpiStyleSelector** | Style router: one `selector` integer plus a list of trigger words, feeding a chain of MpiStyleLoras banks. Outputs an `MPI_STYLE` bundle. |
| **MpiStyleLoras** | A bank of 5 LoRA slots for an MpiStyleSelector — applies only the LoRA matching the selector. Chainable (bank 1 = selector 1-5, bank 2 = 6-10, …), outputs model, clip and prompt. |
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
| **MpiHasAudio** | Check if a video file (by path) contains an audio stream via ffmpeg. Outputs a boolean to gate an audio wire into CreateVideo. |
| **MpiH3Length** | Convert a wanted duration into a valid MiniMax H3 frame count. H3 only generates `n % 17 == 5` frames at 24 fps, so whole seconds are mostly unreachable — asking for 2 s gives 2.33 s, and 8 s is the shortest exact one. Snaps to the **nearest** valid count (core snaps up, which is never closer), and outputs the true `seconds` plus `in_trained_range` (false outside 124–362 frames, where the model runs but was not trained). |
| **MpiH3References** | MiniMax H3 reference conditioning with every slot exposed at once — 9 images, 3 videos with paired soundtracks, 3 standalone audio. Core's node grows its slots one at a time, so a host app compiling a saved workflow would need a pre-authored branch per combination (2¹⁸ of them); this one takes them all and drops the empty ones itself, renumbering the survivors so core's index-based soundtrack pairing still lines up. Empty means nothing connected **or** an Mpi loader with `block_if_empty` off (a 1×1 image / 1-sample waveform) — real black images and real silence pass through. Conditioning is built by core's own `MiniMaxH3ReferenceToVideo`, so it cannot drift. Write the prompt against **slot** numbers — `<Picture 1>` means `ref_image_1` — and the node rewrites them to the ordinals core presents, dropping any tag whose slot is empty. That is what keeps a tag pointing at the same input when a slot ahead of it is left empty, and it is the only way to be right about audio: a reference video's soundtrack consumes an `<Audio j>`, and whether the file *has* one is unknown until it is decoded. Outputs `ref_tags` with the resulting map. |

### Video

| Node | Description |
|---|---|
| **MpiSaveVideo** | Fast save-video node with no in-graph preview and optional audio. Encodes an IMAGE frame batch (+ optional AUDIO) to a single .mp4 in one libx264 pass, on the engine — much faster than CreateVideo+SaveVideo for video export, and remote gens transfer only the final mp4. Toggle audio with the `use_audio` boolean; output length is pinned to the video (short audio padded, long audio trimmed), or flip `truncate_to_audio` to cut the clip to the audio instead. GPU-agnostic (no nvenc). |
| **MpiLoadVideo** | Fast, no-frills video loader by path. Decodes frames + audio and outputs source info (fps, frame_count, duration, width, height, has_audio) in one ffmpeg pass — no in-graph preview, no VHS param surface, so it loads much faster than Load Video (Path). Input named `string` to match MpiString / MpiAnyChecker; empty/missing path blocks downstream, unless block_if_empty is off (then outputs a blank 1x1 image + silent audio). |
| **MpiLoadAudio** | Load audio from a file path into a ComfyUI AUDIO object, like the built-in Load Audio but driven by a `string` path (matches MpiString / MpiAnyChecker). Works on anything ffmpeg reads, including the audio track of a video. Empty/missing/audio-less path blocks downstream, unless block_if_empty is off (then outputs silent audio). |

### Latent

| Node | Description |
|---|---|
| **MpiSaveLatent** | Save a latent to `<output>/latents/<filename>.latent` (or an absolute path), overwriting, then either stop the branch there or carry on — the `boolean` widget is the same continue/block gate as MpiBlocker. Unlike the core Save Latent node it handles packed audio+video latents (MiniMax H3's NestedTensor pair, which has no `.contiguous()` and crashes core). Single-tensor files stay byte-compatible with core Load Latent. The BOOLEAN output is never blocked, so it can drive a second branch. **This is an output node, so ComfyUI runs it on every submit** — set `enabled` false on runs that must not save (a stage-2 continue), or it drags the sampler feeding it along too. |
| **MpiLoadLatent** | Load a latent written by MpiSaveLatent and continue the run — the second half of a two-stage sample. Rebuilds a packed audio+video pair (H3) or a plain tensor. Filename is a plain text field, not a dropdown, so a file written this session needs no UI refresh. A missing file blocks the latent output and reports `loaded` false, so the other branch can generate it instead. |
| **MpiLatentUpscale** | Upscale a latent to a target given in **pixels**, dividing by the model's own VAE `stride` instead of core's hardcoded `// 8`. Two fixes in one node. Core's Upscale Latent crashes on a packed audio+video latent (`AttributeError: 'NestedTensor' object has no attribute 'reshape'`, because `common_upscale` reshapes) — this unbinds the pair, scales only the half with spatial dims (picked by `dim() >= 5`, not by index) and passes the audio half `[B,32,2,T]` through untouched. And core assumes the SD VAE factor, so an H3 target typed in pixels comes out at DOUBLE with no error at all — set `stride` 16 for MiniMax H3 and Krea2, 8 for the SD/SDXL family, and the number you type is the number you get. Built for the hi-res fix: small stage 1, upscale here, let stage 2's low sigmas finish at the target so only half the denoise pays for the resolution. |

### Sampling

| Node | Description |
|---|---|
| **MpiVideoSamplingPreview** | Live RGB sampling previews for video models ComfyUI cannot preview. A format that names no `taesd_decoder_name` — MiniMax H3's `MiniMaxH3Video` / `MiniMaxH3AV` — falls back to `Latent2RGB`, the blocky colour blobs, and it does so *silently*: previews still appear, they are just bad. Wire a tiny **TAEHV** decoder (`taeh3` for H3) into `vae` via **MpiTinyVaeLoader** — a plain VAELoader *cannot* build `taeh3` — and this decodes the in-progress latent for real. Frames stream on the **standard binary preview channel**, preceded by one `VHS_latentpreview` marker, so a host app driving ComfyUI over the websocket receives them with no extra plumbing (unlike node-widget previewers, which base64 onto a private event only the ComfyUI web UI can see). Every sampler step decodes the **whole clip from frame 0** and bursts it, announced at `preview_rate` fps, so the preview plays at the video's own speed instead of jumping one frame per step. Decoding the whole thing is not a cost choice: a TAEHV is *temporal*, its MemBlocks chain state forward, so a window taken out of the middle decodes with cold state and comes out as garbage. H3's own chunking is honoured as well — it codes 17 pixel frames per 5 latent tokens, so each chunk's prefix is trimmed instead of one global trim, and the encoder's 3-token tail pad is dropped, which is what makes the frame count and the timing exact (bit-exact against KJNodes' `TAEHVDecoder` on the same weight). Handles H3's packed audio+video NestedTensor by previewing the video half. A preview failure is caught, reported once and the generation continues. Sits anywhere on the model wire before the sampler and passes the model through untouched. |
| **MpiTinyVaeLoader** | Loads a tiny **TAEHV** preview decoder that ComfyUI's own `VAELoader` cannot build. Core sizes a TAEHV's edge convs as `image_channels * patch_size**2` and only selects `patch_size = 2` for `latent_channels in [48, 32]`. MiniMax H3's `taeh3` is a **24**-channel latent with a **12**-wide decoder (3 RGB x 4 temporal frames), so core builds it 3 wide and `VAELoader` raises `size mismatch for decoder.22.bias: [12] vs [3]`. There is no branch for that shape and no argument that reaches it. This rebuilds the two edge convs at the right width — a strict state-dict load then matches all 128 tensors exactly — and corrects the two spatial ratios and the latent scaling core's fallback branch guessed wrong (`taeh3` is 16x spatial and needs no scaling). Any decoder core already handles is passed straight through to `VAELoader`'s own code path. Feed the output to **MpiVideoSamplingPreview**. |

---

## License

[MIT](https://choosealicense.com/licenses/mit/)
