import { app } from "../../scripts/app.js";

// Paint MPI_PACK wires black so a bundled connection reads as its own thing at
// a glance. Newer frontends take the colour off the canvas map and older ones
// off the static LGraphCanvas one, so set whichever is present.
const PACK_COLOR = "#000000";

app.registerExtension({
  name: "MpiPack",

  setup() {
    if (app.canvas?.default_connection_color_byType) {
      app.canvas.default_connection_color_byType["MPI_PACK"] = PACK_COLOR;
    }
    if (typeof LGraphCanvas !== "undefined" && LGraphCanvas.link_type_colors) {
      LGraphCanvas.link_type_colors["MPI_PACK"] = PACK_COLOR;
    }
  },
});
