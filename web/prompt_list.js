import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MpiPromptList",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MpiPromptList") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      if (origOnNodeCreated) origOnNodeCreated.call(this);

      const widgetSpecific = this.widgets.find(w => w.name === "specific_item");
      const widgetAmount = this.widgets.find(w => w.name === "amount");
      const widgetProb = this.widgets.find(w => w.name === "probability");

      if (!widgetSpecific || !widgetAmount || !widgetProb) return;

      const styleWidget = (widget, grayedOut) => {
        widget.disabled = grayedOut;
        const el = widget.inputEl;
        if (el) {
          el.style.pointerEvents = grayedOut ? "none" : "auto";
          el.style.opacity = grayedOut ? "0.5" : "1";
          el.style.filter = grayedOut ? "grayscale(100%)" : "none";
        }
      };

      const updateGrayedOutState = () => {
        const val = widgetSpecific.value;
        styleWidget(widgetProb, val === 0);
        styleWidget(widgetAmount, val >= 0);
        this.setDirtyCanvas(true);
      };

      widgetSpecific.callback = () => {
        console.log("specific_item changed:", widgetSpecific.value);
        updateGrayedOutState();
      };

      setTimeout(updateGrayedOutState, 50);
    };
  },
});