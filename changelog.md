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
