/**
 * ConfigController - Manages pipeline configuration UI
 *
 * Responsibilities:
 * - Render stages panel matching project detail view
 * - Handle stage selection with expandable options
 * - Manage stage dependencies (e.g., COLMAP enables GSIR/MoCap)
 * - Calculate time estimates
 * - Submit configuration to start processing
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { formatDuration } from '../utils/time.js';
import { ELEMENTS, EVENTS } from '../config/constants.js';

const ALL_STAGES = ['ingest', 'depth', 'roto', 'cleanplate', 'colmap', 'interactive', 'mama', 'mocap', 'gsir', 'camera'];

const STAGE_LABELS = {
    ingest: 'Ingest Video Frames',
    depth: 'Zdepth Estimation',
    roto: 'Auto Segmentation (Roto)',
    cleanplate: 'Clean Plate',
    colmap: 'Camera Tracking (COLMAP)',
    interactive: 'Interactive Segmentation',
    mama: 'Matte Refinement (VideoMaMa)',
    mocap: 'Human MoCap (SMPL+)',
    gsir: '3D Reconstruction (GS-IR)',
    camera: 'Camera Export (Alembic)',
};

const STAGE_OPTIONS = {
    ingest: [
        { id: 'fps', label: 'FPS', type: 'number', default: '', placeholder: 'auto' },
    ],
    roto: [
        { id: 'prompt', label: 'Prompt', type: 'text', default: 'person', placeholder: 'person,car,ball' },
        { id: 'separate_instances', label: 'Separate Instances', type: 'checkbox', default: true },
    ],
    colmap: [
        { id: 'quality', label: 'Quality', type: 'select', default: 'medium', options: ['low', 'medium', 'high', 'slow'] },
        { id: 'dense', label: 'Dense', type: 'checkbox', default: false },
        { id: 'mesh', label: 'Mesh', type: 'checkbox', default: false },
        { id: 'no_masks', label: 'No Masks', type: 'checkbox', default: false },
    ],
    gsir: [
        { id: 'iterations', label: 'Iterations', type: 'number', default: 35000, placeholder: '35000' },
    ],
    camera: [
        { id: 'rotation_order', label: 'Rotation', type: 'select', default: 'zxy', options: ['xyz', 'zxy', 'zyx'] },
    ],
};

const STAGE_DEPENDENCIES = {
    colmap: ['gsir', 'mocap', 'camera'],
    roto: ['mama'],
};

export class ConfigController {
    constructor() {
        this.elements = {
            configForm: dom.getElement(ELEMENTS.CONFIG_FORM),
            configStages: dom.getElement('config-stages'),
            stagesCounter: dom.getElement('config-stages-counter'),
            skipExisting: dom.getElement(ELEMENTS.SKIP_EXISTING),
            timeEstimate: dom.getElement('time-estimate'),
        };

        this.selectedStages = new Set();
        this.stageOptions = {};
        this.disabledStages = new Set(['mama', 'mocap', 'gsir', 'camera']);

        this._boundHandlers = {};
        this.setupEventListeners();
        this.renderStagesPanel();
    }

    setupEventListeners() {
        this._boundHandlers.onFormSubmit = (e) => {
            e.preventDefault();
            this.handleSubmit();
        };
        this._boundHandlers.onStateChange = (e) => {
            if (e.detail.updates?.vramAnalysis !== undefined) {
                this.updateVramWarnings();
            }
        };

        if (this.elements.configForm) {
            this.elements.configForm.addEventListener('submit', this._boundHandlers.onFormSubmit);
        }

        stateManager.addEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);
    }

    renderStagesPanel() {
        if (!this.elements.configStages) return;

        const vramAnalysis = stateManager.get('vramAnalysis');
        const vramStages = vramAnalysis?.stages || {};

        const html = ALL_STAGES.map(stage => {
            const label = STAGE_LABELS[stage] || stage.toUpperCase();
            const isDisabled = this.disabledStages.has(stage);
            const disabledClass = isDisabled ? 'disabled' : '';

            const vramInfo = vramStages[stage];
            const vramStatus = vramInfo?.status || 'ok';
            const vramMessage = vramInfo?.message || '';
            const vramGb = vramInfo?.base_vram_gb;

            let vramDisplay = '';
            let vramStatusClass = '';
            let vramTitle = vramMessage;

            if (vramGb !== undefined && vramGb > 0) {
                if (vramStatus === 'ok') {
                    vramDisplay = `${vramGb} GB`;
                } else if (vramStatus === 'warning' || vramStatus === 'insufficient') {
                    vramDisplay = `⚠️ ${vramGb} GB`;
                    vramStatusClass = `vram-${vramStatus}`;
                } else if (vramStatus === 'chunked') {
                    vramDisplay = `⏳ ${vramGb} GB`;
                    vramStatusClass = 'vram-chunked';
                }
            }

            const optionsHtml = this.renderStageOptionsHtml(stage);
            const hasOptions = STAGE_OPTIONS[stage] && STAGE_OPTIONS[stage].length > 0;
            const expandIcon = hasOptions ? '<span class="stage-expand-icon">▸</span>' : '';

            const safeStage = dom.escapeHTML(stage);
            return `
            <div class="stage-wrapper ${disabledClass}" data-stage="${safeStage}">
                <div class="stage-status-item selectable ${disabledClass}" data-stage="${safeStage}">
                    ${expandIcon}
                    <div class="stage-marker"></div>
                    <span class="stage-status-name">${dom.escapeHTML(label)}</span>
                    <span class="stage-vram ${vramStatusClass}" title="${dom.escapeHTML(vramTitle)}">${vramDisplay}</span>
                </div>
                ${optionsHtml}
            </div>
        `;
        }).join('');

        dom.setHTML(this.elements.configStages, html);
        this.bindStageClickHandlers();
        this.updateStagesCounter();
    }

    renderStageOptionsHtml(stage) {
        const options = STAGE_OPTIONS[stage];
        if (!options || options.length === 0) return '';

        const optionInputs = options.map(opt => {
            const currentValue = this.stageOptions[stage]?.[opt.id] ?? opt.default;
            const safeLabel = dom.escapeHTML(opt.label);

            if (opt.type === 'checkbox') {
                const checked = currentValue ? 'checked' : '';
                return `
                    <label class="stage-option-item">
                        <input type="checkbox" data-stage="${dom.escapeHTML(stage)}" data-option="${dom.escapeHTML(opt.id)}" ${checked}>
                        <span>${safeLabel}</span>
                    </label>`;
            } else if (opt.type === 'select') {
                const optionsHtml = opt.options.map(o => {
                    const selected = o === currentValue ? 'selected' : '';
                    return `<option value="${dom.escapeHTML(o)}" ${selected}>${dom.escapeHTML(o)}</option>`;
                }).join('');
                return `
                    <label class="stage-option-item">
                        <span>${safeLabel}</span>
                        <select data-stage="${dom.escapeHTML(stage)}" data-option="${dom.escapeHTML(opt.id)}">${optionsHtml}</select>
                    </label>`;
            } else {
                const placeholder = opt.placeholder ? `placeholder="${dom.escapeHTML(opt.placeholder)}"` : '';
                const value = currentValue !== '' ? `value="${dom.escapeHTML(String(currentValue))}"` : '';
                return `
                    <label class="stage-option-item">
                        <span>${safeLabel}</span>
                        <input type="${opt.type}" data-stage="${dom.escapeHTML(stage)}" data-option="${dom.escapeHTML(opt.id)}" ${value} ${placeholder}>
                    </label>`;
            }
        }).join('');

        return `<div class="stage-options-panel hidden">${optionInputs}</div>`;
    }

    bindStageClickHandlers() {
        const stageItems = this.elements.configStages.querySelectorAll('.stage-status-item');
        stageItems.forEach(item => {
            item.addEventListener('click', () => this.toggleStageSelection(item));
        });
    }

    toggleStageSelection(item) {
        const stage = item.dataset.stage;
        const wrapper = item.closest('.stage-wrapper');

        if (this.disabledStages.has(stage)) {
            return;
        }

        const optionsPanel = wrapper?.querySelector('.stage-options-panel');
        const expandIcon = item.querySelector('.stage-expand-icon');

        if (this.selectedStages.has(stage)) {
            this.selectedStages.delete(stage);
            item.classList.remove('selected');
            if (wrapper) wrapper.classList.remove('selected');
            if (optionsPanel) optionsPanel.classList.add('hidden');
            if (expandIcon) expandIcon.textContent = '▸';

            this.handleStageDeselected(stage);
        } else {
            this.selectedStages.add(stage);
            item.classList.add('selected');
            if (wrapper) wrapper.classList.add('selected');
            if (optionsPanel) {
                optionsPanel.classList.remove('hidden');
                this.bindStageOptionHandlers(stage, optionsPanel);
            }
            if (expandIcon) expandIcon.textContent = '▾';

            this.handleStageSelected(stage);
        }

        this.updateStagesCounter();
        this.updateTimeEstimate();
    }

    handleStageSelected(stage) {
        const dependents = STAGE_DEPENDENCIES[stage];
        if (dependents) {
            dependents.forEach(dep => {
                this.disabledStages.delete(dep);
                const wrapper = this.elements.configStages.querySelector(`.stage-wrapper[data-stage="${dep}"]`);
                const item = wrapper?.querySelector('.stage-status-item');
                if (wrapper) wrapper.classList.remove('disabled');
                if (item) item.classList.remove('disabled');
            });
        }
    }

    handleStageDeselected(stage) {
        const dependents = STAGE_DEPENDENCIES[stage];
        if (dependents) {
            dependents.forEach(dep => {
                this.disabledStages.add(dep);
                this.selectedStages.delete(dep);
                const wrapper = this.elements.configStages.querySelector(`.stage-wrapper[data-stage="${dep}"]`);
                const item = wrapper?.querySelector('.stage-status-item');
                if (wrapper) {
                    wrapper.classList.add('disabled');
                    wrapper.classList.remove('selected');
                }
                if (item) {
                    item.classList.add('disabled');
                    item.classList.remove('selected');
                }
                const optionsPanel = wrapper?.querySelector('.stage-options-panel');
                if (optionsPanel) optionsPanel.classList.add('hidden');
            });
        }
    }

    bindStageOptionHandlers(stage, panel) {
        const inputs = panel.querySelectorAll('input, select');
        inputs.forEach(input => {
            if (input.dataset.bound) return;
            input.dataset.bound = 'true';

            input.addEventListener('change', () => {
                if (!this.stageOptions[stage]) {
                    this.stageOptions[stage] = {};
                }
                if (input.type === 'checkbox') {
                    this.stageOptions[stage][input.dataset.option] = input.checked;
                } else if (input.type === 'number') {
                    this.stageOptions[stage][input.dataset.option] = input.value ? Number(input.value) : null;
                } else {
                    this.stageOptions[stage][input.dataset.option] = input.value;
                }
            });

            input.addEventListener('click', (e) => e.stopPropagation());
        });
    }

    updateStagesCounter() {
        if (this.elements.stagesCounter) {
            dom.setText(this.elements.stagesCounter, `${this.selectedStages.size}/${ALL_STAGES.length}`);
        }
    }

    updateVramWarnings() {
        this.renderStagesPanel();

        this.selectedStages.forEach(stage => {
            const wrapper = this.elements.configStages.querySelector(`.stage-wrapper[data-stage="${stage}"]`);
            const item = wrapper?.querySelector('.stage-status-item');
            if (wrapper) wrapper.classList.add('selected');
            if (item) item.classList.add('selected');
            const optionsPanel = wrapper?.querySelector('.stage-options-panel');
            const expandIcon = item?.querySelector('.stage-expand-icon');
            if (optionsPanel) {
                optionsPanel.classList.remove('hidden');
                this.bindStageOptionHandlers(stage, optionsPanel);
            }
            if (expandIcon) expandIcon.textContent = '▾';
        });
    }

    updateTimeEstimate() {
        const config = stateManager.get('config');
        const videoInfo = stateManager.get('videoInfo');

        if (!config || !videoInfo) {
            if (this.elements.timeEstimate) {
                dom.setText(this.elements.timeEstimate, '--');
            }
            return;
        }

        const frameCount = videoInfo.frame_count;

        if (!frameCount || typeof frameCount !== 'number' || frameCount <= 0) {
            if (this.elements.timeEstimate) {
                dom.setText(this.elements.timeEstimate, 'Unknown');
            }
            return;
        }

        let totalSeconds = 0;
        this.selectedStages.forEach(stageId => {
            const stage = config.stages?.[stageId];
            if (stage) {
                const timePerFrame = stage.estimatedTimePerFrame || 0;
                totalSeconds += timePerFrame * frameCount;
            }
        });

        if (this.elements.timeEstimate) {
            const formatted = totalSeconds > 0 ? `~${formatDuration(totalSeconds)}` : '--';
            dom.setText(this.elements.timeEstimate, formatted);
        }
    }

    getSelectedStages() {
        return Array.from(this.selectedStages);
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
            skip_existing: this.elements.skipExisting?.checked || false,
            stage_options: {},
        };

        selectedStages.forEach(stage => {
            const opts = this.stageOptions[stage] || {};
            const defaults = STAGE_OPTIONS[stage] || [];
            const stageOpts = {};

            defaults.forEach(def => {
                const value = opts[def.id] !== undefined ? opts[def.id] : def.default;
                if (value !== '' && value !== null) {
                    stageOpts[def.id] = value;
                }
            });

            if (Object.keys(stageOpts).length > 0) {
                config.stage_options[stage] = stageOpts;
            }
        });

        if (config.stage_options.roto?.prompt) {
            config.roto_prompt = config.stage_options.roto.prompt;
        }

        try {
            await apiService.startProcessing(projectId, config);

            const projectName = stateManager.get('projectName');
            stateManager.startProcessing(projectId, projectName, selectedStages);

            dom.hide(this.elements.configForm);
        } catch (error) {
            stateManager.showError(error.message || 'Failed to start processing');
        }
    }

    reset() {
        this.selectedStages.clear();
        this.stageOptions = {};
        this.disabledStages = new Set(['mama', 'mocap', 'gsir', 'camera']);
        dom.hide(this.elements.configForm);
        this.renderStagesPanel();
    }

    destroy() {
        if (this.elements.configForm && this._boundHandlers.onFormSubmit) {
            this.elements.configForm.removeEventListener('submit', this._boundHandlers.onFormSubmit);
        }

        if (this._boundHandlers.onStateChange) {
            stateManager.removeEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);
        }

        this._boundHandlers = {};
    }
}
