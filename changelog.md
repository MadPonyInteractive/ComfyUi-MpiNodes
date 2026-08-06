# Version: V1.0.5
- MpiClearVram - added
- MpiExpoFloat - added
- Descriptions to nodes missing them
- fixed: MpiGridDimensions now avoids repeated tiles when fed to UltimateSDUpscale
# Version: V1.0.6
- MpiLogger - added
# Version: V1.0.7
- MpiListCount - added
# Version: V1.0.8
- MpiListRange - added
# Version: V1.1.0
- fixed: MpiMaskDebugInfo now has a display name (was showing raw class name)
- fixed: removed orphan MpiChoiceSelector / MpiPresetSelector display entries (no backing class)
- MpiHasAudio - added
# Version: V1.1.1
- MpiUpscaleModelScale - added
# Version: V1.1.2
- MpiReroute - added
- MpiConditioningReroute - added
# Version: V1.1.3
- MpiSaveVideo - added: fast save-video node, no in-graph preview, optional audio (use_audio toggle) — faster video export for certain workflows
# Version: V1.1.4
- server route: added POST /mpi/reload-extra-paths — re-reads extra_model_paths.yaml at runtime so a model folder added mid-session registers without restarting ComfyUI (Cubric Vision MPI-219)
# Version: V1.1.5
- fixed: MpiSaveVideo now writes the filename_prefix (was saving `_00001.mp4` with no prefix — used the wrong return value from get_save_image_path)
- fixed: MpiSaveVideo now encodes audio from VHS/LazyAudioMap sources (was checking isinstance(dict), which a LazyAudioMap fails, so no audio track was ever written)
# Version: V1.1.6

- MpiBlockIfEmpty - added
- MpiAnyChecker - added
- MpiLoadImageFromPath - added
- MpiSeedPassthrough - added
- MpiLoadVideo - added
- MpiLoadAudio - added
- MpiCrop - added
- fixed: MpiSaveVideo crops odd dimensions to even (libx264/yuv420p refused e.g. 2899x806) and streams frames to ffmpeg stdin instead of buffering the whole raw video in RAM (was doubling multi-GB clips, maxing out memory)
# Version: V1.1.7
- fixed: MpiMath no longer uses eval() — replaced with an ast-based arithmetic evaluator (registry bans eval/exec as an RCE risk, which had been silently flagging every version since 1.0.5). Same math behavior, no code-execution surface.
# Version: V1.2.0
- MpiLoadImageFromPath - added a channel combo (alpha/red/green/blue) to pick which channel the mask output comes from
- MpiLoadVideo - added a has_audio BOOLEAN output (True when the file has an audio track); reuses the probe already run during audio load, so no extra cost
# Version: V1.2.1
- fixed: MpiMath expression parser now uses ast module-mode parsing, removing the last dynamic-execution keyword from the source (the registry scanner was still flagging on the leftover token). No behavior change.
# Version: V1.2.2
- MpiIsListEmpty - added
- MpiBlockIfEmptyList - added
- MpiBlocker - added
- MpiBlockIfEmpty - class renamed to MpiAnyBlocker (matches its "Mpi Any Blocker" display name); old class name kept as an alias so existing saved workflows still load
- fixed: MpiMath now supports ternary (`b if a == 1 else 0.0`) expressions again — the ast evaluator was missing IfExp support that eval() had for free, so every conditional expression errored
- fixed: MpiGetImageAtIndex blocks downstream on an empty image batch instead of throwing IndexError
- MpiExecLogger - added
# Version: V1.2.3
- fixed: MpiSaveVideo no longer hangs forever when saving with audio — stderr is now drained on a thread so ffmpeg cannot deadlock on a full stderr pipe while frames stream to stdin
- fixed: MpiSaveVideo output length is now pinned to the video — shorter audio is padded with silence, longer audio is trimmed, so it no longer either cut the video to a sub-second clip or freeze on the last frame while long audio played out
- MpiSaveVideo - added a truncate_to_audio toggle (off = keep full video and pad audio; on = cut the clip to the audio length). Grayed out when use_audio is off.
# Version: V1.2.4
- MpiCombo - removed (BETA; superseded by a vanilla ComfyUI node)
- MpiComboMaker - removed (BETA; only fed MpiCombo)
- MpiCombo1 through MpiCombo10 - removed (txt-file-backed, unmaintained)
- removed web/combos.js — the only frontend network call in the pack (a same-origin fetch of combos.json); likely the registry scanner's python_network_operations false-positive trigger
# Version: V1.2.5

- MpiMaskSquareBbox - added
- MpiLoadImageFromPath - added block_if_empty toggle (off outputs a blank 1x1 image instead of blocking)
- MpiLoadVideo - added block_if_empty toggle (off outputs a blank image + silent audio instead of blocking)
- MpiLoadAudio - added block_if_empty toggle (off outputs silent audio instead of blocking)
- MpiScaledDimensions - added scaled_image output (image resized to the computed dimensions) + upscale_method input

# Version: V1.2.6
- MpiLogger - added message input and mode dropdown ("value" logs the input value, "message" logs only the message when the input arrives)
- MpiBox - added (MPI_BOX type: width/height/x/y rectangle, x/y = top-left)
- MpiFromBox - added (unpacks MPI_BOX into width, height, x, y)
- MpiBoxCrop - added (crops an image to an MPI_BOX, outputs image + clamped box)
- MpiBoxMask - added (mask the size of the image with the MPI_BOX drawn as a white rectangle, outputs mask + clamped box)

# Version: V1.2.7
- MpiInpaintHeal - added (matches an inpainted region's colour and grain to a ring of untouched pixels around the mask, fixing the colour drift and flat texture generative fills leave behind)
- MpiStyleSelector - added (one selector int + a trigger-words list drives a chain of LoRA banks; replaces the per-lora MpiMath gating workaround)
- MpiStyleLoras - added (bank of 5 LoRA slots for MpiStyleSelector, chainable - each bank claims the next 5 selector values; outputs model, clip and prompt)
- MpiClamp - added (clamps an int or float between a min and a max, preserves int inputs)
- MpiTextContains - added (true if any listed word or phrase appears in the input text as a whole word - multiline box separated by commas or new lines, case-insensitive, plural-aware and spacing-agnostic; whitelist/blacklist check)
- MpiAnySwitch10 - added (same as MpiAnySwitch with 10 inputs instead of 5)
- MpiCompare - added b_value float widget used when the b input is not connected, converted to the type of a (0/1 for booleans, truncated for ints); raises if a is not a number, connect b for that; b is now optional
- MpiPacker - added (MPI_PACK type: bundles up to 5 values of any type into one wire, packs can nest)
- MpiUnpacker - added (unpacks an MPI_PACK into 5 outputs in slot order, empty slots block downstream)
- MpiPacker10 - added (same as MpiPacker with 10 slots)
- MpiUnpacker10 - added (same as MpiUnpacker with 10 outputs)
- MpiComparePacks - added (compares two packs slot by slot, tensor-safe and recursive; outputs equal + first_diff)
- MpiSaveLatent - added (saves a latent then optionally blocks the branch; handles the packed audio+video NestedTensor pair that crashes core SaveLatent)
- MpiLoadLatent - added (loads a latent back and continues a two-stage sample; rebuilds the audio+video pair, blocks + reports false when the file is missing)
- MpiBooleanInvert - added (inverts a boolean; wire-only input, no widget)
- MpiBlocker - input is now lazy: blocking skips the upstream work feeding it, not just the downstream branch
- MpiSaveLatent - added enabled input (lazy samples): off skips the save AND the work feeding it, so an output node no longer forces its upstream sampler to run
- MpiH3Length - added (wanted duration -> valid MiniMax H3 frame count on the 17k+5 grid, nearest not up; outputs true seconds and in_trained_range)
- MpiSaveLatent - now reports the saved file as `ui.latents` the way core SaveLatent does (filename + subfolder + type), so a host app driving a two-stage run can read it back from /history and fetch it over /view; the save was previously invisible outside ComfyUI
- MpiLoadLatent - now looks in the engine `input/` folder first and falls back to `<output>/latents/`, so a latent staged there by a host app resolves; a hand-run graph is unaffected because nothing stages into `input/`
