import { app } from "../../scripts/app.js";

// Gray out truncate_to_audio when use_audio is off — the toggle only affects
// how the audio track is length-matched, so it's meaningless without audio.
app.registerExtension({
  name: "MpiSaveVideo",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "MpiSaveVideo") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      if (origOnNodeCreated) origOnNodeCreated.call(this);

      const useAudio = this.widgets.find(w => w.name === "use_audio");
      const truncate = this.widgets.find(w => w.name === "truncate_to_audio");
      if (!useAudio || !truncate) return;

      const update = () => {
        truncate.disabled = !useAudio.value;
        this.setDirtyCanvas(true);
      };

      const origCallback = useAudio.callback;
      useAudio.callback = function () {
        if (origCallback) origCallback.apply(this, arguments);
        update();
      };

      setTimeout(update, 50);
    };
  }
});
