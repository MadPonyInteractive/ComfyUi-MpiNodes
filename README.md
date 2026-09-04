# ComfyUi-MpiNodes

A ComfyUI custom node pack by [Mad Pony Interactive](https://github.com/MadPonyInteractive) with over 100 utilities for logic, math, prompt generation, image operations, model management, and workflow automation.

[![Registry](https://img.shields.io/badge/Comfy%20Registry-mad--pony--interactive-blue)](https://registry.comfy.org/publishers/mad-pony-interactive/nodes/ComfyUi-MpiNodes)

**These nodes power [Cubric Vision](https://github.com/MadPonyInteractive/Cubric-Vision)** — a free,
open-source desktop app for AI image and video generation. Vision installs and manages ComfyUI,
the models and this pack for you, and wraps them in a real editing UI: gallery, masking,
inpainting, painting, cropping and project management. If you found this pack through one of our
workflows, the app is where those workflows come from.


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
| **MpiMaskSquareBbox** | Square bounding box around a mask, centered on the mask and clamped (shrunk if needed) to stay fully inside the image. Outputs a filled square MASK plus x, y, and side length. Optional padding around the tight box. On a batch — a video mask — the box is the **union of every frame** and the square is returned **for every frame**, so the region holds still for the whole clip instead of crawling, and a video consumer gets as many mask frames as it was given. |
| **MpiBox** | Build an `MPI_BOX` rectangle from width/height/x/y, where x/y are the **top-left** corner. One wire carries a region into any box-aware node instead of four loose INT sockets. Consumers clamp the box to the image, so out-of-bounds values are safe — and x/y may be **negative**, so a box can start outside the image to sit tight on a subject at the frame edge. |
| **MpiFromBox** | Unpack an `MPI_BOX` back into width, height, x, y integers — the escape hatch for feeding raw INTs to other nodes. |
| **MpiBoxCrop** | Crop an image to an `MPI_BOX` region. Outputs the cropped image plus the clamped box actually used. A box fully outside the image passes the image through unchanged. `pad` (optional, off by default) pads an overhanging box back out to its requested size by replicating the edge pixels, so the crop keeps the aspect that was asked for — what a square reference crop needs. |
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
| **MpiAudioRange** | Cut a soundtrack to a **frame** range, using the same inclusive `start`/`end` that `MpiListRange` takes — so one pair of numbers windows the picture and the sound together. Exists for windowed work on an audio-video model: cutting the frames and leaving the audio whole hands the model a soundtrack that does not line up with the picture, and nothing downstream can notice — every tensor is valid, the result is just conditioned on the wrong moment. Wire the loader's own `fps`; a guessed rate slides the audio rather than failing. |
| **MpiImageSplice** | Write a run of frames back into a longer clip at a given frame — the return leg of windowed work: cut a window out, run the expensive thing on it alone, splice the result back. Pairs with `MpiH3DecodeAV`, whose output outside the mask is already the original pixel, so the window boundary is exact and needs no blending. A mismatched canvas, or a patch running past the end, raises — either one otherwise lands as a clip that plays but is wrong. |
| **MpiAudioSplice** | Write a run of audio back into a longer soundtrack at a **frame** offset — the sound half of what `MpiImageSplice` does for picture, and the inverse `MpiAudioRange` shipped without. Takes the same inclusive `start`, so the number that cut a window writes it back. Windowed work on an audio-video model cuts both streams and has to splice both back; without this the regenerated soundtrack stays window-length while the picture comes back full-length, which reads as a video whose sound stops early rather than as a wiring mistake — VHS writes the short track without complaint. The crossfade is `MpiH3DecodeAV`'s own `splice_audio`, so there is one implementation of the equal-power fade rather than two that can drift. A patch running past the end raises and says by how many samples, naming the two things that actually cause it: a start that names a different window, or a patch from a different clip. A mono patch is expanded to a stereo track instead of broadcasting into a crash, and a cut spliced straight back is a bit-exact no-op. |
| **MpiHasAudio** | Check if a video file (by path) contains an audio stream via ffmpeg. Outputs a boolean to gate an audio wire into CreateVideo. |
| **MpiH3Length** | Convert a wanted duration into a valid MiniMax H3 frame count. H3 only generates `n % 17 == 5` frames at 24 fps, so whole seconds are mostly unreachable — asking for 2 s gives 2.33 s, and 8 s is the shortest exact one. Snaps to the **nearest** valid count (core snaps up, which is never closer), and outputs the true `seconds` plus `in_trained_range` (false outside 124–362 frames, where the model runs but was not trained). |
| **MpiH3ImageToVideo** | MiniMax H3 image-to-video conditioning that tolerates EMPTY frame inputs, so one graph covers t2va, first-frame, last-frame and first+last instead of a pre-authored branch per combination. Core's `MiniMaxH3ImageToVideo` already skips a keyframe that arrives as `None`, but a graph cannot send `None` down a connected link, so a host app compiling a saved workflow had to carry four copies of the node behind a lattice of booleans. Empty means nothing connected **or** an Mpi loader with `block_if_empty` off (a 1×1 image) — a genuinely black first frame reports its true size and passes through untouched. The conditioning and the AV latent are built by core's own node, so the tensor maths cannot drift. Both frames are cover-cropped to the canvas first, because core is asymmetric about it — it stretches `first_frame` (`crop='disabled'`) and cover-crops `last_frame` (`'center'`), so an off-aspect source came back squashed and the two keyframes disagreed with each other. Crop, never pad: letterbox bars baked into frame 0 get animated as scenery. Also what makes a SECOND conditioning at upscaled dimensions practical, which is what an H3 latent-upscale refine pass needs: the keyframe latent is encoded at the first stage's width/height and core sizes its cond rows off the target grid, so an upscaled latent otherwise raises a shape mismatch in `SamplerCustomAdvanced`. |
| **MpiH3References** | MiniMax H3 reference conditioning with every slot exposed at once — 9 images, 3 videos with paired soundtracks, 3 standalone audio. Core's node grows its slots one at a time, so a host app compiling a saved workflow would need a pre-authored branch per combination (2¹⁸ of them); this one takes them all and drops the empty ones itself, renumbering the survivors so core's index-based soundtrack pairing still lines up. Empty means nothing connected **or** an Mpi loader with `block_if_empty` off (a 1×1 image / 1-sample waveform) — real black images and real silence pass through. Conditioning is built by core's own `MiniMaxH3ReferenceToVideo`, so it cannot drift. Write the prompt against **slot** numbers — `<Picture 1>` means `ref_image_1` — and the node rewrites them to the ordinals core presents, dropping any tag whose slot is empty. That is what keeps a tag pointing at the same input when a slot ahead of it is left empty, and it is the only way to be right about audio: a reference video's soundtrack consumes an `<Audio j>`, and whether the file *has* one is unknown until it is decoded. Outputs `ref_tags` with the resulting map. |
| **MpiH3EncodeAV** | Encode a clip **and its soundtrack** into one MiniMax H3 AV latent. Core encodes the two streams separately (`VAEEncode` + `VAEEncodeAudio`) and offers no way to join them, so a joint latent — the thing `MpiH3MaskedPrefix` takes as its context — was previously only reachable through a third-party fork that monkey-patches two ComfyUI internals at import time. Resamples the soundtrack to the audio VAE's own rate when it differs. Batch 1 only, matching H3 itself: a second item would pair the wrong soundtrack to the picture rather than fail. With the optional `mask` it is also the **inpainting** encode: a per-frame mask (from SAM3, or any per-frame source) is packed onto the video latent's own grid by union — not interpolated, which on H3's period-5 packing snaps each latent step to a single picked frame and makes a short mask vanish silently — and the audio half is masked all-keep, sized off the encoded audio latent so no frame count or frame rate has to be supplied. The mask may cover the whole clip **or just a range of it**, which is what a segmenter run over part of the clip returns: `mask_start` says which clip frame the mask's first frame lands on, and a mask that runs past the end raises rather than inpainting the wrong moment. This is what lets a host app reach H3 inpainting from IMAGE + AUDIO + MASK, with no `VIDEO` type and no file loader in the graph. When a mask is connected, a clip whose frame count is off H3's 17k+5 grid **raises at encode**, naming the nearest valid count, instead of surfacing as a mask overhang once a full sample has already been paid for — a 6-frame clip packs to 5 and the VAE drops the sixth without complaint. `audio_start` / `audio_end` open a window in the **soundtrack** the way the mask opens one in the picture, and `audio_ranges` (`0-10, 40-50, 90-100`) opens several disjoint ones in a single pass — the capability LanPaint's interactive editor has and no core node exposes, spelled as text because nothing emits the 1-D `[F]` MASK LanPaint consumes. This is context-aware foley, and it is the thing a standalone foley model cannot do: that generates in isolation from a silent clip, while H3 hears the rest of the track as unmasked context, so the window matches the room, the mic and the ambience, and is in sync by construction because one model produces both halves. Ranges are given in **frames** and snapped outwards to a multiple of 3 — audio runs on a 40 Hz clock against 24 fps picture, so a frame boundary is a whole audio step only every third frame, and an off-grid edge raises nowhere, it just drifts the sound against the picture — then merged where they touch, since two ranges three frames apart can collide once snapped. An audio window with no picture mask regenerates sound over untouched video, which is the pure foley case. The limitation is structural: an audio mask selects a **time range, not a source**, because audio sums every source into the same samples, so the window regenerates dialogue, room tone and footsteps together — `MpiH3DecodeAV`'s `audio_mode` is the layering answer. A second `info` output reports what was actually masked on both halves, including how many audio steps were left unmasked, and says so explicitly when the answer is none: a clip whose audio is entirely masked has nothing to match and returns the model's own score at its own level. Pair with `MpiH3DecodeAV`. |
| **MpiH3DecodeAV** | Decode the video half of an H3 AV latent and composite it back into the original frames through the same mask, which is what finishes an `MpiH3EncodeAV` inpaint. Outside the mask the result is the **original pixel**, not a VAE round trip of it, so an inpaint does not quietly soften the whole frame; `feather` ramps the mask edge so the seam does not show — dilate then Gaussian, matching LanPaint's `blend_overlap` to float noise in the interior, so the ramp sits **outside** the mask you drew and the inpainted content keeps full strength at its own edge. It diverges from LanPaint in one place on purpose: replicate padding, so a mask touching the frame border (a subject walking out of shot) is not faded away there — LanPaint's zero padding takes it to 0.58. It now finishes the **audio** half of the job too, through a second `AUDIO` output: connect `audio_vae` + `audio` and the same `audio_start` / `audio_end` / `audio_ranges` the encode was given. It shipped without them deliberately — nothing was spliced into the soundtrack, so there was no audio seam to hide — and that held exactly as long as the encode masked audio all-keep, which an audio window ends. Outside the window the sample is the **original**, matching what the picture half already promised. The decode is taken raw rather than through core's `vae_decode_audio`, which divides by its own standard deviation and would land the window in a different loudness domain from the track it is being spliced into. `audio_crossfade` is feather's counterpart, and its curve follows the mode because the two are not the same operation: `replace` crosses between two uncorrelated takes of one moment, where equal power holds the level, while `mix` fades a layer in over an original that keeps playing, where equal power would start the layer at 71% on the very first sample — a click of exactly the kind the crossfade exists to remove. `audio_mode` `mix` sums the window **under** the original instead of replacing it, which is the only way to layer a sound onto a track that already carries a performance: the mask selects a time range and not a source, so `replace` throws the original dialogue away along with the silence it was meant to fill. `audio_gain` (default 1.0) is the trim for the level the model decided on, which is rarely the level of the track it has to sit in; around 0.3 is where a new layer sits under an existing track rather than fighting it. With no `audio_vae`, or `audio_end` at 0, the audio input passes straight through, so a picture-only inpaint keeps its old wiring and its old result. `mask_start` / `mask_end` must match the encode's. A latent whose frame count does not match the originals raises, naming the `17k+5` grid as the cause. `mask_mode` chooses which mask the composite runs through: `per-frame` is the exact mask you drew, and `as sampled` rebuilds the coarser one the model was actually given — the latent grid, the union of the frames each latent step packs, and the DiT's own patch pooling. The painted region is always wider than the drawn one (measured +27% on a 124-frame 1536x640 clip, where the model sees 48x20 mask cells), so compositing per-frame keeps a thin slice of a wide repaint and puts the seam INSIDE the region the model worked to. `as sampled` moves the boundary out to where the model stopped, which is blockier but lets feather come right down. |
| **MpiH3MaskedPrefix** | Continue a MiniMax H3 clip **without regenerating its tail**. Writes the prior clip's encoded frames into the front of the target latent and masks that region out of sampling, so the head is preserved exactly (measured at PSNR 38 dB against the source — a VAE round trip) and there is nothing to trim afterwards. `context_frames` snaps **down** to 39 / 90 / 141 …, the only lengths that sit on H3's `17k+5` video grid *and* divide by 3 so audio's 40 Hz clock also lands on a whole step; an off-grid value does not raise, it drifts the audio against the picture and starts the continuation from a moment that never happened, so the snap is reported rather than assumed. The nested video+audio noise mask is built at the latent's own resolution deliberately — core's `reshape_mask` interpolates a mask that does not match, smearing the one boundary that has to stay hard. **A clean prefix is not an anchor:** on its own the model preserves those frames and then renders an unrelated scene, so pair it with a single frame-0 `MiniMaxH3AddGuide`. |

### Video

| Node | Description |
|---|---|
| **MpiSaveVideo** | Fast save-video node with no in-graph preview and optional audio. Encodes an IMAGE frame batch (+ optional AUDIO) to a single .mp4 in one libx264 pass, on the engine — much faster than CreateVideo+SaveVideo for video export, and remote gens transfer only the final mp4. Toggle audio with the `use_audio` boolean; output length is pinned to the video (short audio padded, long audio trimmed), or flip `truncate_to_audio` to cut the clip to the audio instead. GPU-agnostic (no nvenc). |
| **MpiLoadVideo** | Fast, no-frills video loader by path. Decodes frames + audio and outputs source info (fps, frame_count, duration, width, height, has_audio) in one ffmpeg pass — no in-graph preview, no VHS param surface, so it loads much faster than Load Video (Path). Input named `string` to match MpiString / MpiAnyChecker; empty/missing path blocks downstream, unless block_if_empty is off (then outputs a blank 1x1 image + silent audio). Optional `force_rate` resamples to a target frame rate inside that same decode pass (0 = source rate), so fps, frame_count and duration all come out at the forced rate. |
| **MpiLoadAudio** | Load audio from a file path into a ComfyUI AUDIO object, like the built-in Load Audio but driven by a `string` path (matches MpiString / MpiAnyChecker). Works on anything ffmpeg reads, including the audio track of a video. Empty/missing/audio-less path blocks downstream, unless block_if_empty is off (then outputs silent audio). |

### Latent

| Node | Description |
|---|---|
| **MpiSaveLatent** | Save a latent to `<output>/latents/<filename>.latent` (or an absolute path), overwriting, then either stop the branch there or carry on — the `boolean` widget is the same continue/block gate as MpiBlocker. Unlike the core Save Latent node it handles packed audio+video latents (MiniMax H3's NestedTensor pair, which has no `.contiguous()` and crashes core). Single-tensor files stay byte-compatible with core Load Latent. The BOOLEAN output is never blocked, so it can drive a second branch. **This is an output node, so ComfyUI runs it on every submit** — set `enabled` false on runs that must not save (a stage-2 continue), or it drags the sampler feeding it along too. |
| **MpiLoadLatent** | Load a latent written by MpiSaveLatent and continue the run — the second half of a two-stage sample. Rebuilds a packed audio+video pair (H3) or a plain tensor. Filename is a plain text field, not a dropdown, so a file written this session needs no UI refresh. A missing file blocks the latent output and reports `loaded` false, so the other branch can generate it instead. |
| **MpiStageLatents** | The two-stage latent handshake in ONE node: saves stage 1, gates the preview, and loads the latent back on a continue. `is_continue` / `is_preview` are **widgets**, so a host app drives both stages of a single workflow file instead of shipping a `_stage2` twin — a continue requests no latent inputs, so the stage-1 sampler is genuinely skipped rather than run and discarded. A missing latent **raises** instead of blocking every branch, which would otherwise leave a silent no-output run. Handles the packed audio+video latents (MiniMax H3) that crash core Save/Load Latent. |
| **MpiLatentUpscale** | Upscale a latent to a target given in **pixels**, dividing by the model's own VAE `stride` instead of core's hardcoded `// 8`. Two fixes in one node. Core's Upscale Latent crashes on a packed audio+video latent (`AttributeError: 'NestedTensor' object has no attribute 'reshape'`, because `common_upscale` reshapes) — this unbinds the pair, scales only the half with spatial dims (picked by `dim() >= 5`, not by index) and passes the audio half `[B,32,2,T]` through untouched. And core assumes the SD VAE factor, so an H3 target typed in pixels comes out at DOUBLE with no error at all — set `stride` 16 for MiniMax H3 and Krea2, 8 for the SD/SDXL family, and the number you type is the number you get. Built for the hi-res fix: small stage 1, upscale here, let stage 2's low sigmas finish at the target so only half the denoise pays for the resolution. |

### Sampling

| Node | Description |
|---|---|
| **MpiVideoSamplingPreview** | Live RGB sampling previews for video models ComfyUI cannot preview. A format that names no `taesd_decoder_name` — MiniMax H3's `MiniMaxH3Video` / `MiniMaxH3AV` — falls back to `Latent2RGB`, the blocky colour blobs, and it does so *silently*: previews still appear, they are just bad. Wire a tiny **TAEHV** decoder (`taeh3` for H3) into `vae` via **MpiTinyVaeLoader** — a plain VAELoader *cannot* build `taeh3` — and this decodes the in-progress latent for real. Frames stream on the **standard binary preview channel**, preceded by one `VHS_latentpreview` marker, so a host app driving ComfyUI over the websocket receives them with no extra plumbing (unlike node-widget previewers, which base64 onto a private event only the ComfyUI web UI can see). Every sampler step decodes the **whole clip from frame 0** and bursts it, announced at `preview_rate` fps, so the preview plays at the video's own speed instead of jumping one frame per step. Decoding the whole thing is not a cost choice: a TAEHV is *temporal*, its MemBlocks chain state forward, so a window taken out of the middle decodes with cold state and comes out as garbage. H3's own chunking is honoured as well — it codes 17 pixel frames per 5 latent tokens, so each chunk's prefix is trimmed instead of one global trim, and the encoder's 3-token tail pad is dropped, which is what makes the frame count and the timing exact (bit-exact against KJNodes' `TAEHVDecoder` on the same weight). Handles H3's packed audio+video NestedTensor by previewing the video half. On LTX, `LanPaint`-style guide latents (`LTXVAddGuide`'s `keyframe_idxs`) are trimmed off the tail before decoding, so the preview shows the clip and not the conditioning frames pasted after it; H3 has no guide node, which is why the previewer shipped without it. A preview failure is caught, reported once and the generation continues. Sits anywhere on the model wire before the sampler and passes the model through untouched. |
| **MpiTinyVaeLoader** | Loads a tiny **TAEHV** preview decoder that ComfyUI's own `VAELoader` cannot build. Core sizes a TAEHV's edge convs as `image_channels * patch_size**2` and only selects `patch_size = 2` for `latent_channels in [48, 32]`. MiniMax H3's `taeh3` is a **24**-channel latent with a **12**-wide decoder (3 RGB x 4 temporal frames), so core builds it 3 wide and `VAELoader` raises `size mismatch for decoder.22.bias: [12] vs [3]`. There is no branch for that shape and no argument that reaches it. This rebuilds the two edge convs at the right width — a strict state-dict load then matches all 128 tensors exactly — and corrects the two spatial ratios and the latent scaling core's fallback branch guessed wrong (`taeh3` is 16x spatial and needs no scaling). Any decoder core already handles is passed straight through to `VAELoader`'s own code path. Feed the output to **MpiVideoSamplingPreview**. |

### Gaussian Splats

| Node | Description |
|---|---|
| **MpiBrushTrain** | Trains a Gaussian splat from a COLMAP dataset with **Brush** (Apache-2.0) and returns the exported `.ply` path. Brush is a native binary: it is downloaded per-platform on first use and **SHA-256 verified against a pinned checksum** — pinned rather than fetched, because a checksum served from the same host as the payload proves nothing about that host. Set `brush_path` to point at a copy you manage yourself and the node never touches the network. Two measured quirks shape the node. Brush **writes zero bytes to stdout when it is not on a TTY**, so there is no step line to parse; progress is read from the export directory instead, where Brush drops `export_{iter}.ply` every `--export-every` steps — a silent run is a working run, not a hung one. And a SplatKit dataset carries **four** COLMAP models, two of them under `_spheresfm_work/` on camera model 11 (SPHERE), which Brush picks between nondeterministically and then dies on with `Invalid camera model`; so the node stages a root holding exactly one model, hardlinking the images rather than copying tens of GB. Cancellable — a 30000-step bake takes tens of minutes, and interrupting the prompt kills the trainer. `max_resolution` (default 2048) is the third quirk and the one that costs RAM: Brush caches one decoded u8 RGB copy of every training view in HOST memory, so the bake needs `N_views x min(face_size, max_resolution)^2 x 3` bytes — ~11.5 GB for 984 faces at 2048, 4.5 GB at 1280 — and its own default of 1920 quietly discards the top of anything rendered larger. |

---

## License

[GNU AGPL-3.0](LICENSE) — the same licence as Cubric Vision, and compatible with
ComfyUI's GPL-3.0 (GPLv3 §13).

Free to use, modify and share. If you build on this pack, your work carries the same
licence — including when it is offered to others over a network.

> Versions up to and including 1.2.6 were published under MIT. That grant stands for
> those versions; 1.2.7 onward is AGPL-3.0.
