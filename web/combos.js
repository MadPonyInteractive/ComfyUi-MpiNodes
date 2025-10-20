import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "MpiCombo",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MpiCombo") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      if (origOnNodeCreated) origOnNodeCreated.call(this);


      this.setup = async () => {
        const combo_name = this.widgets.find(w => w.name === "combo_name");
        const options = this.widgets.find(w => w.name === "options");

        if (!combo_name || !options) return;

        const updateOptions = async () => {
          try {
            // const res = await fetch("extensions/ComfyUi-MpiNodes/user/combos.json");
            const res = await fetch(`extensions/ComfyUi-MpiNodes/user/combos.json?cacheBust=${Date.now()}`);
            const data = await res.json();

            const key = combo_name.value;
            const newOptions = Array.isArray(data[key]) ? data[key] : ["None"];

            options.options.values = newOptions;
            options.value = newOptions.includes(options.value) ? options.value : newOptions[0];

            options.inputEl?.dispatchEvent(new Event("change", { bubbles: true }));
            this.setDirtyCanvas(true);
          } catch (err) {
          }
        };

        combo_name.callback = () => {
          updateOptions();
        };

        updateOptions(); // run once on setup
      };

      // ✅ Manually invoke setup to ensure it runs
      this.setup?.();
    };
  }
});