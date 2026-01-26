/**
 * ComfyUI Auto-Load Extension
 *
 * This extension automatically loads a workflow from /input/auto_load_workflow.json
 * when ComfyUI starts. Used by launch_interactive_segmentation.py for Docker mode.
 */

import { app } from "../../scripts/app.js";

const AUTO_LOAD_PATH = "/input/auto_load_workflow.json";

app.registerExtension({
    name: "vfx.autoload",

    async setup() {
        // Wait for app to be fully initialized
        await new Promise(resolve => setTimeout(resolve, 1500));

        try {
            const response = await fetch(AUTO_LOAD_PATH);
            if (!response.ok) {
                console.log("[VFX AutoLoad] No auto-load workflow found at", AUTO_LOAD_PATH);
                return;
            }

            const workflow = await response.json();
            console.log("[VFX AutoLoad] Loading workflow from", AUTO_LOAD_PATH);

            await app.loadGraphData(workflow);
            console.log("[VFX AutoLoad] Workflow loaded successfully!");

        } catch (error) {
            console.log("[VFX AutoLoad] Could not auto-load workflow:", error.message);
        }
    }
});
