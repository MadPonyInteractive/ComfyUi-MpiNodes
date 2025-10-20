// import { app } from "../../scripts/app.js";

// app.registerExtension({
//   name: "MpiDummyToggleNode",

//   async beforeRegisterNodeDef(nodeType, nodeData, app) {
//     if (nodeData.name !== "MpiDummyToggleNode") return;

//     const origOnNodeCreated = nodeType.prototype.onNodeCreated;

//     nodeType.prototype.onNodeCreated = function () {
//       if (origOnNodeCreated) origOnNodeCreated.call(this);

//       const widgetEnable = this.widgets.find(w => w.name === "enable_extra");
//       const widgetExtra = this.widgets.find(w => w.name === "extra_value");

//       if (!widgetEnable || !widgetExtra) return;

//       const inputName = "extra_value";
//       const inputType = "FLOAT"; // Adjust to match your Python type

//       const updateVisibility = () => {
//         const visible = widgetEnable.value;

//         const inputIndex = this.findInputSlot(inputName);

//         if (!visible && inputIndex !== -1) {
//           this.removeInput(inputIndex);
//         }

//         if (visible && inputIndex === -1) {
//           this.addInput(inputName, inputType);
//         }

//         this.setSize([this.size[0], 0]);
//         this.setDirtyCanvas(true);
//         app.canvas.draw(true, true);
//       };

//       widgetEnable.callback = () => {
//         console.log("enable_extra toggled:", widgetEnable.value);
//         updateVisibility();
//       };

//       setTimeout(() => {
//         updateVisibility();
//       }, 50);
//     };
//   }
// });


import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MpiDummyToggleNode",

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name !== "MpiDummyToggleNode") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      if (origOnNodeCreated) origOnNodeCreated.call(this);

      const widgetEnable = this.widgets.find(w => w.name === "enable_extra");
      const widgetExtra = this.widgets.find(w => w.name === "extra_value");

      if (!widgetEnable || !widgetExtra) return;

      const updateVisibility = () => {
        const visible = widgetEnable.value;
        widgetExtra.hidden = !visible;

        this.setSize([this.size[0], 0]);
        this.setDirtyCanvas(true);
        app.canvas.draw(true, true);
      };

      // Bind callback early
      widgetEnable.callback = () => {
        console.log("enable_extra toggled:", widgetEnable.value);
        updateVisibility();
      };

      // Run once after full mount
      this.setup = async () => {
        updateVisibility();
      };
    };
  }
});
