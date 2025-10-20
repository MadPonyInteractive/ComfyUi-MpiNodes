import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MpiDynamicCombo",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MpiDynamicCombo") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      if (origOnNodeCreated) origOnNodeCreated.call(this);

      const fields = [
        this.widgets.find(w => w.name === "option_1"),
        this.widgets.find(w => w.name === "option_2"),
        this.widgets.find(w => w.name === "option_3"),
        this.widgets.find(w => w.name === "option_4"),
        this.widgets.find(w => w.name === "option_5"),
        this.widgets.find(w => w.name === "option_6"),
        this.widgets.find(w => w.name === "option_7"),
        this.widgets.find(w => w.name === "option_8"),
      ];

      const combo = this.widgets.find(w => w.name === "selected");
      if (!combo) return;

      const updateComboChoices = () => {
        const choices = fields
          .map(w => w?.value?.trim())
          .filter(v => v); // remove empty

        const newChoices = choices.length ? choices : ["None"];
        const currentValue = combo.value;

        combo.options.values = newChoices;
        combo.value = newChoices.includes(currentValue) ? currentValue : newChoices[0];

        combo.inputEl?.dispatchEvent(new Event("change", { bubbles: true }));
        this.setDirtyCanvas(true);
      };

      // Bind callbacks early
      fields.forEach(w => {
        if (w) w.callback = updateComboChoices;
      });

      // Run once after full mount
      this.setup = async () => {
        updateComboChoices();
      };
    };
  }
});

// import { app } from "../../scripts/app.js";

// app.registerExtension({
//   name: "MpiDynamicCombo",

//   async beforeRegisterNodeDef(nodeType, nodeData) {
//     if (nodeData.name !== "MpiDynamicCombo") return;

//     const origOnNodeCreated = nodeType.prototype.onNodeCreated;

//     nodeType.prototype.onNodeCreated = function () {
//       if (origOnNodeCreated) origOnNodeCreated.call(this);

//       const fields = [
//         this.widgets.find(w => w.name === "option_1"),
//         this.widgets.find(w => w.name === "option_2"),
//         this.widgets.find(w => w.name === "option_3"),
//         this.widgets.find(w => w.name === "option_4"),
//         this.widgets.find(w => w.name === "option_5"),
//         this.widgets.find(w => w.name === "option_6"),
//         this.widgets.find(w => w.name === "option_7"),
//         this.widgets.find(w => w.name === "option_8"),
//       ];

//       const combo = this.widgets.find(w => w.name === "selected");
//       if (!combo) return;

//       const updateComboChoices = () => {
//         const choices = fields
//           .map(w => w?.value?.trim())
//           .filter(v => v); // remove empty

//         const newChoices = choices.length ? choices : ["None"];
//         const currentValue = combo.value;

//         combo.options.values = newChoices;
//         combo.value = newChoices.includes(currentValue) ? currentValue : newChoices[0];

//         combo.inputEl?.dispatchEvent(new Event("change", { bubbles: true }));
//         this.setDirtyCanvas(true);
//       };

//       fields.forEach(w => {
//         if (w) w.callback = updateComboChoices;
//       });

//       setTimeout(updateComboChoices, 50);
//     };
//   }
// });