import { app } from "../../scripts/app.js";

const TYPE = "MiniMaxH3DLSSNRRuntimeAuditT8Advanced";
const PROBES = new Set(["static_only", "feature_probe_1_frame"]);

// Remove only the known retired widget; keep mode and GPU selections intact.
// Never mutate the stored source graph or interpret the old checkbox as consent.
function migrate(info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length !== 5 ||
        typeof values[1] !== "boolean" || !PROBES.has(values[2])) return info;
    const result = { ...info, widgets_values: [values[0], ...values.slice(2)] };
    if (info.widgets_values_named) {
        result.widgets_values_named = { ...info.widgets_values_named };
        delete result.widgets_values_named.accept_external_runtime_license;
    }
    return result;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.dlss-runtime-widget-migration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (info, ...args) {
            return configure.call(this, migrate(info), ...args);
        };
    },
});
