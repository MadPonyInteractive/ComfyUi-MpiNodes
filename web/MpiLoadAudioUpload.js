import { app } from "../../scripts/app.js";

// The core frontend adds its upload button to any node whose REQUIRED `audio`
// combo carries audio_upload, but the audio player that button drives
// (`audioUI`) is added only to core's own audio nodes, and the button breaks
// without it. Declare the same player for MpiLoadAudioUpload, ordered before
// `upload` because widgets are built in that order. Declarations only: the
// upload itself is done by the core frontend.
app.registerExtension({
  name: "MpiLoadAudioUpload",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MpiLoadAudioUpload") return;
    const { upload, audioUI, ...rest } = nodeData.input.required;
    nodeData.input.required = {
      ...rest,
      audioUI: ["AUDIO_UI", {}],
      upload: upload ?? ["AUDIOUPLOAD", {}],
    };
  },
});
