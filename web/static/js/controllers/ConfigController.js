/**
 * ConfigController - Manages pipeline configuration UI
 *
 * Responsibilities:
 * - Handle stage checkbox changes
 * - Apply presets
 * - Manage stage dependencies (e.g., COLMAP enables GSIR/MoCap)
 * - Calculate time estimates
 * - Submit configuration to start processing
 *
 * Follows Single Responsibility Principle by only handling config UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { formatDuration } from '../utils/time.js';
import { ELEMENTS, EVENTS } from '../config/constants.js';

export class ConfigController {
    constructor() {
        this.elements = {
            configForm: dom.getElement(ELEMENTS.CONFIG_FORM),
            rotoPrompt: dom.getElement(ELEMENTS.ROTO_PROMPT),
            rotoPromptWrapper: dom.getElement('roto-prompt-wrapper'),
            stageRoto: dom.getElement('stage-roto'),
            skipExisting: dom.getElement(ELEMENTS.SKIP_EXISTING),
            timeEstimate: dom.getElement('time-estimate'),
            presetButtons: dom.getElements('.preset-btn'),
            stageCheckboxes: dom.getElements('input[name="stage"]'),
        };

        this.setupEventListeners();
        this.applyDefaultPreset();
    }

    setupEventListeners() {
        // Form submission
        if (this.elements.configForm) {
            this.elements.configForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleSubmit();
            });
        }

        // Preset buttons
        this.elements.presetButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                this.applyPreset(preset);
            });
        });

        // Stage checkboxes
        this.elements.stageCheckboxes.forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                this.handleStageChange(checkbox);
                this.updateTimeEstimate();
            });
        });

        // Roto checkbox - toggle prompt visibility
        if (this.elements.stageRoto) {
            this.elements.stageRoto.addEventListener('change', () => {
                this.toggleRotoPrompt();
            });
        }

        // Update VRAM warnings when vramAnalysis becomes available
        stateManager.addEventListener(EVENTS.STATE_CHANGED, (e) => {
            if (e.detail.updates?.vramAnalysis !== undefined) {
                this.updateVramWarnings();
            }
        });
    }

    applyDefaultPreset() {
        // Apply "full" preset by default
        this.applyPreset('full');
    }

    applyPreset(presetName) {
        const config = stateManager.get('config');
        if (!config) return;

        const preset = config.presets?.[presetName];
        if (!preset) return;

        // Uncheck all checkboxes first
        this.elements.stageCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
        });

        // Check stages in preset
        preset.stages.forEach(stageId => {
            const checkbox = document.querySelector(`input[name="stage"][value="${stageId}"]`);
            if (checkbox && !checkbox.disabled) {
                checkbox.checked = true;
            }
        });

        // Update active preset button
        this.elements.presetButtons.forEach(btn => {
            if (btn.dataset.preset === presetName) {
                dom.addClass(btn, 'active');
            } else {
                dom.removeClass(btn, 'active');
            }
        });

        // Update dependencies, time estimate, and VRAM warnings
        this.updateStageDependencies();
        this.updateTimeEstimate();
        this.toggleRotoPrompt();
        this.updateVramWarnings();
    }

    setDependentStage(stageValue, enabled) {
        const checkbox = document.querySelector(`input[value="${stageValue}"]`);
        if (!checkbox) return;

        if (enabled) {
            checkbox.disabled = false;
        } else {
            checkbox.disabled = true;
            checkbox.checked = false;
        }
    }

    handleStageChange(checkbox) {
        const stage = checkbox.value;

        if (stage === 'colmap') {
            const enabled = checkbox.checked;
            this.setDependentStage('gsir', enabled);
            this.setDependentStage('mocap', enabled);
            this.setDependentStage('camera', enabled);
        }

        if (stage === 'roto') {
            this.setDependentStage('mama', checkbox.checked);
        }

        this.elements.presetButtons.forEach(btn => {
            dom.removeClass(btn, 'active');
        });
    }

    updateStageDependencies() {
        const colmapChecked = document.querySelector('input[value="colmap"]')?.checked;
        this.setDependentStage('gsir', colmapChecked);
        this.setDependentStage('mocap', colmapChecked);
        this.setDependentStage('camera', colmapChecked);

        const rotoChecked = document.querySelector('input[value="roto"]')?.checked;
        this.setDependentStage('mama', rotoChecked);
    }

    updateVramWarnings() {
        const vramAnalysis = stateManager.get('vramAnalysis');
        if (!vramAnalysis?.stages) return;

        this.elements.stageCheckboxes.forEach(checkbox => {
            const stage = checkbox.value;
            const analysis = vramAnalysis.stages[stage];
            if (!analysis) return;

            const label = checkbox.closest('label');
            if (!label) return;

            let warningEl = label.querySelector('.vram-warning');

            if (analysis.status === 'warning' || analysis.status === 'insufficient') {
                if (!warningEl) {
                    warningEl = document.createElement('span');
                    warningEl.title = analysis.message;
                    warningEl.textContent = ' \u26A0\uFE0F';
                    const textEl = label.querySelector('.checkbox-text, .stage-title, span');
                    if (textEl) {
                        textEl.appendChild(warningEl);
                    }
                }
                const statusClass = analysis.status === 'insufficient' ? 'vram-insufficient' : 'vram-warning-status';
                warningEl.className = `vram-warning ${statusClass}`;
                warningEl.title = analysis.message;
            } else if (analysis.status === 'chunked') {
                if (!warningEl) {
                    warningEl = document.createElement('span');
                    warningEl.className = 'vram-warning vram-chunked';
                    warningEl.title = analysis.message;
                    warningEl.textContent = ' \u23F3';
                    const textEl = label.querySelector('.checkbox-text, .stage-title, span');
                    if (textEl) {
                        textEl.appendChild(warningEl);
                    }
                }
                warningEl.title = analysis.message;
            } else if (warningEl) {
                warningEl.remove();
            }
        });
    }

    toggleRotoPrompt() {
        if (!this.elements.rotoPromptWrapper) return;

        if (this.elements.stageRoto?.checked) {
            dom.show(this.elements.rotoPromptWrapper);
        } else {
            dom.hide(this.elements.rotoPromptWrapper);
        }
    }

    updateTimeEstimate() {
        const config = stateManager.get('config');
        const videoInfo = stateManager.get('videoInfo');

        if (!config || !videoInfo) {
            // No config or video info yet
            if (this.elements.timeEstimate) {
                dom.setText(this.elements.timeEstimate, '--');
            }
            return;
        }

        const selectedStages = this.getSelectedStages();
        const frameCount = videoInfo.frame_count;

        // Validate frame count exists
        if (!frameCount || typeof frameCount !== 'number' || frameCount <= 0) {
            if (this.elements.timeEstimate) {
                dom.setText(this.elements.timeEstimate, 'Unknown');
            }
            return;
        }

        // Calculate estimate
        let totalSeconds = 0;
        selectedStages.forEach(stageId => {
            const stage = config.stages?.[stageId];
            if (stage) {
                const timePerFrame = stage.estimatedTimePerFrame || 0;
                totalSeconds += timePerFrame * frameCount;
            }
        });

        // Display estimate
        if (this.elements.timeEstimate) {
            const formatted = totalSeconds > 0 ? `~${formatDuration(totalSeconds)}` : '--';
            dom.setText(this.elements.timeEstimate, formatted);
        }
    }

    getSelectedStages() {
        const stages = [];
        this.elements.stageCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                stages.push(checkbox.value);
            }
        });
        return stages;
    }

    async handleSubmit() {
        const selectedStages = this.getSelectedStages();

        if (selectedStages.length === 0) {
            stateManager.showError('Please select at least one processing stage');
            return;
        }

        const projectId = stateManager.get('projectId');
        if (!projectId) {
            stateManager.showError('No project loaded');
            return;
        }

        const config = {
            stages: selectedStages,
            roto_prompt: this.elements.rotoPrompt?.value || 'person',
            skip_existing: this.elements.skipExisting?.checked || false,
        };

        try {
            // Start processing
            await apiService.startProcessing(projectId, config);

            // Update state
            const projectName = stateManager.get('projectName');
            stateManager.startProcessing(projectId, projectName, selectedStages);

            // Hide config form
            dom.hide(this.elements.configForm);

            // Processing panel will be shown by ProcessingController
        } catch (error) {
            stateManager.showError(error.message || 'Failed to start processing');
        }
    }

    reset() {
        this.elements.stageCheckboxes.forEach(checkbox => {
            checkbox.checked = false;
            checkbox.disabled = false;
        });

        this.setDependentStage('gsir', false);
        this.setDependentStage('mocap', false);
        this.setDependentStage('camera', false);
        this.setDependentStage('mama', false);

        if (this.elements.rotoPrompt) {
            this.elements.rotoPrompt.value = 'person';
        }
        if (this.elements.skipExisting) {
            this.elements.skipExisting.checked = false;
        }

        dom.hide(this.elements.configForm);
        this.applyDefaultPreset();
    }
}
