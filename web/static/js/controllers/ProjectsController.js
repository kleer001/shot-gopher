/**
 * ProjectsController - Manages projects list and detail panel
 *
 * Responsibilities:
 * - Load and display projects list
 * - Handle project selection
 * - Display project details (stages, VRAM, actions)
 * - Auto-refresh projects
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { ELEMENTS, EVENTS, CSS_CLASSES } from '../config/constants.js';

const ALL_STAGES = ['ingest', 'depth', 'roto', 'cleanplate', 'colmap', 'interactive', 'mama', 'mocap', 'gsir', 'camera'];

const STAGE_OUTPUT_DIRS = {
    ingest: 'source',
    depth: 'depth',
    roto: 'roto',
    cleanplate: 'cleanplate',
    colmap: 'colmap',
    interactive: 'roto',
    mama: 'matte',
    mocap: 'mocap',
    gsir: 'gsir',
    camera: 'camera',
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
};

function formatFileSize(bytes) {
    if (bytes === null || bytes === undefined || bytes === 0) {
        return '0 B';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let unitIndex = 0;
    let size = bytes;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    if (unitIndex === 0) {
        return `${Math.round(size)} B`;
    }
    if (size >= 100) {
        return `${Math.round(size)} ${units[unitIndex]}`;
    }
    if (size >= 10) {
        return `${size.toFixed(1)} ${units[unitIndex]}`;
    }
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

export class ProjectsController {
    constructor() {
        this.elements = {
            projectsList: dom.getElement(ELEMENTS.PROJECTS_LIST),
            detailPanel: dom.getElement(ELEMENTS.PROJECT_DETAIL_PANEL),
            detailProjectName: dom.getElement(ELEMENTS.DETAIL_PROJECT_NAME),
            detailStages: dom.getElement(ELEMENTS.DETAIL_STAGES),
            stagesCounter: dom.getElement(ELEMENTS.STAGES_COUNTER),
            vramInfoSection: dom.getElement(ELEMENTS.VRAM_INFO_SECTION),
            vramInfo: dom.getElement(ELEMENTS.VRAM_INFO),
            openFolderBtn: dom.getElement(ELEMENTS.OPEN_FOLDER_BTN),
            deleteProjectBtn: dom.getElement(ELEMENTS.DELETE_PROJECT_BTN),
            reprocessBtn: dom.getElement(ELEMENTS.REPROCESS_BTN),
            interactiveCompleteBtn: dom.getElement(ELEMENTS.INTERACTIVE_COMPLETE_BTN),
            videoInfoSection: dom.getElement('video-info-section'),
            videoResolution: dom.getElement('video-resolution'),
            videoFrames: dom.getElement('video-frames'),
            videoFps: dom.getElement('video-fps'),
            videoDuration: dom.getElement('video-duration'),
        };

        this.selectedProjectId = null;
        this.selectedStages = new Set();
        this.stageOptions = {};
        this.currentVramData = null;
        this.refreshInterval = null;
        this.processingPollInterval = null;

        this._boundHandlers = {};
        this.setupEventListeners();
        this.bindDetailActions();
        this.loadProjects();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        this._boundHandlers.onUploadComplete = (e) => this.handleUploadComplete(e.detail);
        stateManager.addEventListener(EVENTS.UPLOAD_COMPLETE, this._boundHandlers.onUploadComplete);
    }

    async handleUploadComplete(detail) {
        const projectId = detail.projectId;
        await this.loadProjects();
        await this.selectProjectById(projectId);
    }

    async selectProjectById(projectId) {
        this.selectedProjectId = projectId;
        this.selectedStages.clear();
        this.stageOptions = {};

        const items = this.elements.projectsList?.querySelectorAll('.project-item');
        items?.forEach(el => {
            if (el.dataset.id === projectId) {
                el.classList.add('selected');
            } else {
                el.classList.remove('selected');
            }
        });

        await this.showProjectDetails(projectId);
    }

    bindDetailActions() {
        if (this.elements.openFolderBtn) {
            this.elements.openFolderBtn.addEventListener('click', () => this.openFolder());
        }
        if (this.elements.deleteProjectBtn) {
            this.elements.deleteProjectBtn.addEventListener('click', () => this.deleteProject());
        }
        if (this.elements.reprocessBtn) {
            this.elements.reprocessBtn.addEventListener('click', () => this.reprocessStages());
        }
        if (this.elements.interactiveCompleteBtn) {
            this.elements.interactiveCompleteBtn.addEventListener('click', () => this.completeInteractive());
        }
    }

    async loadProjects() {
        if (!this.elements.projectsList) return;

        try {
            const data = await apiService.getProjects();

            if (data.projects && data.projects.length > 0) {
                this.displayProjects(data.projects.slice(0, 20));
            } else {
                dom.setHTML(this.elements.projectsList, '<p class="no-projects">(none yet)</p>');
            }
        } catch (error) {
            console.error('Failed to load projects:', error);
            dom.setHTML(this.elements.projectsList, '<p class="no-projects">Failed to load</p>');
        }
    }

    displayProjects(projects) {
        const html = projects.map(proj => {
            const projectId = proj.name;
            const safeName = dom.escapeHTML(projectId);
            const safeId = dom.escapeHTML(projectId);
            const sizeDisplay = formatFileSize(proj.size_bytes);
            const selectedClass = this.selectedProjectId === projectId ? 'selected' : '';
            return `
            <div class="project-item ${selectedClass}" data-id="${safeId}">
                <span class="project-name">${safeName}</span>
                <span class="project-size">${sizeDisplay}</span>
            </div>
        `;
        }).join('');

        dom.setHTML(this.elements.projectsList, html);

        const projectItems = this.elements.projectsList.querySelectorAll('.project-item');
        projectItems.forEach(item => {
            item.addEventListener('click', () => {
                this.handleProjectClick(item);
            });
        });

        stateManager.setState({ projects });
    }

    async handleProjectClick(item) {
        const projectId = item.dataset.id;

        document.querySelectorAll('.project-item').forEach(el => el.classList.remove('selected'));
        item.classList.add('selected');

        this.selectedProjectId = projectId;
        this.selectedStages.clear();
        this.stageOptions = {};

        await this.showProjectDetails(projectId);
    }

    async showProjectDetails(projectId) {
        if (!this.elements.detailPanel) return;

        dom.removeClass(this.elements.detailPanel, CSS_CLASSES.HIDDEN);
        this.hideInteractiveCompleteButton();

        if (this.elements.detailProjectName) {
            this.elements.detailProjectName.textContent = projectId;
        }

        try {
            const [projectData, outputsData, vramData, videoInfoData] = await Promise.all([
                apiService.getProject(projectId).catch(() => null),
                apiService.getProjectOutputs(projectId).catch(() => null),
                apiService.getProjectVram(projectId).catch(() => null),
                apiService.getProjectVideoInfo(projectId).catch(() => null),
            ]);

            this.currentVramData = vramData;
            this.renderStagesStatus(projectData, outputsData, vramData);
            this.updateVramDisplay();
            this.renderVideoInfo(videoInfoData);
        } catch (error) {
            console.error('Failed to load project details:', error);
        }
    }

    renderVideoInfo(videoInfoData) {
        const videoInfo = videoInfoData?.video_info;

        if (!videoInfo || Object.keys(videoInfo).length === 0) {
            dom.addClass(this.elements.videoInfoSection, CSS_CLASSES.HIDDEN);
            return;
        }

        dom.removeClass(this.elements.videoInfoSection, CSS_CLASSES.HIDDEN);

        if (videoInfo.resolution && Array.isArray(videoInfo.resolution)) {
            dom.setText(this.elements.videoResolution, `${videoInfo.resolution[0]}x${videoInfo.resolution[1]}`);
        } else {
            dom.setText(this.elements.videoResolution, '--');
        }

        if (videoInfo.frame_count !== undefined && videoInfo.frame_count > 0) {
            if (videoInfo.frame_start !== undefined && videoInfo.frame_end !== undefined) {
                dom.setText(this.elements.videoFrames,
                    `${videoInfo.frame_start} - ${videoInfo.frame_end} (${videoInfo.frame_count.toLocaleString()})`);
            } else {
                dom.setText(this.elements.videoFrames, videoInfo.frame_count.toLocaleString());
            }
        } else {
            dom.setText(this.elements.videoFrames, '--');
        }

        if (videoInfo.fps !== undefined && videoInfo.fps > 0) {
            dom.setText(this.elements.videoFps, videoInfo.fps.toFixed(2));
        } else {
            dom.setText(this.elements.videoFps, '--');
        }

        if (videoInfo.duration !== undefined && videoInfo.duration > 0) {
            const mins = Math.floor(videoInfo.duration / 60);
            const secs = (videoInfo.duration % 60).toFixed(1);
            dom.setText(this.elements.videoDuration, `${mins}:${secs.padStart(4, '0')}`);
        } else if (videoInfo.frame_count && videoInfo.fps) {
            const duration = videoInfo.frame_count / videoInfo.fps;
            const mins = Math.floor(duration / 60);
            const secs = (duration % 60).toFixed(1);
            dom.setText(this.elements.videoDuration, `${mins}:${secs.padStart(4, '0')}`);
        } else {
            dom.setText(this.elements.videoDuration, '--');
        }
    }

    renderStagesStatus(projectData, outputsData, vramData) {
        if (!this.elements.detailStages) return;

        const outputs = outputsData?.outputs || {};
        const vramStages = vramData?.analysis?.stages || {};

        const stageLabels = {
            ingest: 'Ingest',
            depth: 'ZDepth',
            roto: 'Auto Roto',
            cleanplate: 'Clean Plate',
            colmap: 'Camera Tracking',
            interactive: 'Interactive Roto',
            mama: 'Roto Refinement',
            mocap: 'Human Mocap',
            gsir: 'Scene Capture',
            camera: 'Camera Export',
        };

        let completedCount = 0;

        const html = ALL_STAGES.map(stage => {
            const outputDir = STAGE_OUTPUT_DIRS[stage];
            const outputData = outputs[outputDir];
            const hasFiles = outputData && outputData.count > 0;
            const isCompleted = hasFiles;

            if (isCompleted) completedCount++;

            const stageClass = isCompleted ? 'completed' : '';
            const fileCount = outputData?.total_files || outputData?.count || 0;
            const label = stageLabels[stage] || stage.toUpperCase();

            let fileCountDisplay = '';
            let fileSizeDisplay = '';
            if (isCompleted && fileCount > 0) {
                const totalBytes = (outputData?.files || []).reduce((sum, f) => sum + (f.size || 0), 0);
                fileCountDisplay = `${fileCount}`;
                fileSizeDisplay = formatFileSize(totalBytes);
            }

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

            const safeStage = dom.escapeHTML(stage);
            const optionsHtml = this.renderStageOptionsHtml(stage);
            return `
            <div class="stage-wrapper" data-stage="${safeStage}">
                <div class="stage-status-item selectable ${stageClass}" data-stage="${safeStage}">
                    <div class="stage-marker"></div>
                    <span class="stage-status-name">${dom.escapeHTML(label)}</span>
                    <span class="stage-vram ${vramStatusClass}" title="${dom.escapeHTML(vramTitle)}">${vramDisplay}</span>
                    <span class="stage-file-count">${fileCountDisplay}</span>
                    <span class="stage-file-size">${fileSizeDisplay}</span>
                </div>
                ${optionsHtml}
            </div>
        `;
        }).join('');

        dom.setHTML(this.elements.detailStages, html);

        if (this.elements.stagesCounter) {
            dom.setText(this.elements.stagesCounter, `${completedCount}/${ALL_STAGES.length}`);
        }

        const stageItems = this.elements.detailStages.querySelectorAll('.stage-status-item');
        stageItems.forEach(item => {
            item.addEventListener('click', () => this.toggleStageSelection(item));
        });
    }

    toggleStageSelection(item) {
        const stage = item.dataset.stage;
        const wrapper = item.closest('.stage-wrapper');
        const optionsPanel = wrapper?.querySelector('.stage-options-panel');

        if (this.selectedStages.has(stage)) {
            this.selectedStages.delete(stage);
            item.classList.remove('selected');
            if (wrapper) wrapper.classList.remove('selected');
            if (optionsPanel) optionsPanel.classList.add('hidden');
        } else {
            this.selectedStages.add(stage);
            item.classList.add('selected');
            if (wrapper) wrapper.classList.add('selected');
            if (optionsPanel) {
                optionsPanel.classList.remove('hidden');
                this.bindStageOptionHandlers(stage, optionsPanel);
            }
        }

        this.updateProcessButton();
        this.updateVramDisplay();
    }

    renderStageOptionsHtml(stage) {
        const options = STAGE_OPTIONS[stage];
        if (!options || options.length === 0) return '';

        const optionInputs = options.map(opt => {
            const currentValue = this.stageOptions[stage]?.[opt.id] ?? opt.default;
            const safeId = dom.escapeHTML(`${stage}-${opt.id}`);
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

    updateProcessButton() {
        if (!this.elements.reprocessBtn) return;

        const count = this.selectedStages.size;
        if (count > 0) {
            this.elements.reprocessBtn.textContent = `PROCESS ${count} STAGE${count > 1 ? 'S' : ''}`;
            this.elements.reprocessBtn.disabled = false;
        } else {
            this.elements.reprocessBtn.textContent = 'SELECT STAGES TO PROCESS';
            this.elements.reprocessBtn.disabled = true;
        }
    }

    updateVramDisplay() {
        if (!this.elements.vramInfoSection || !this.elements.vramInfo) return;

        const analysis = this.currentVramData?.analysis;
        if (!analysis) {
            dom.addClass(this.elements.vramInfoSection, CSS_CLASSES.HIDDEN);
            return;
        }

        const stages = analysis.stages || {};
        const availableGb = (analysis.available_vram_mb || 0) / 1024;

        if (this.selectedStages.size === 0) {
            dom.addClass(this.elements.vramInfoSection, CSS_CLASSES.HIDDEN);
            return;
        }

        let maxVramGb = 0;
        let hasChunking = false;

        this.selectedStages.forEach(stage => {
            const stageInfo = stages[stage];
            if (stageInfo) {
                const stageVramGb = stageInfo.base_vram_gb || 0;
                if (stageVramGb > maxVramGb) {
                    maxVramGb = stageVramGb;
                }
                if (stageInfo.status === 'chunked') {
                    hasChunking = true;
                }
            }
        });

        dom.removeClass(this.elements.vramInfoSection, CSS_CLASSES.HIDDEN);

        const hasWarning = maxVramGb > availableGb;
        const valueClass = hasWarning ? 'danger' : 'ok';

        let html = `
            <div class="vram-row">
                <span class="vram-label">Required</span>
                <span class="vram-value ${valueClass}">${maxVramGb.toFixed(1)} GB</span>
            </div>
            <div class="vram-row">
                <span class="vram-label">Available</span>
                <span class="vram-value">${availableGb.toFixed(1)} GB</span>
            </div>
        `;

        if (hasChunking) {
            html += `
            <div class="vram-note">
                Chunked processing may be required
            </div>
        `;
        }

        dom.setHTML(this.elements.vramInfo, html);
    }

    async openFolder() {
        if (!this.selectedProjectId) return;

        try {
            await apiService.openProjectFolder(this.selectedProjectId);
        } catch (error) {
            console.error('Failed to open folder:', error);
        }
    }

    async deleteProject() {
        if (!this.selectedProjectId) return;

        const confirmed = confirm(`Delete project "${this.selectedProjectId}"? This cannot be undone.`);
        if (!confirmed) return;

        try {
            await apiService.deleteProject(this.selectedProjectId);
            this.hideDetailPanel();
            this.selectedProjectId = null;
            await this.loadProjects();
        } catch (error) {
            console.error('Failed to delete project:', error);
            alert(`Failed to delete: ${error.message}`);
        }
    }

    async reprocessStages() {
        if (!this.selectedProjectId || this.selectedStages.size === 0) return;

        const stages = Array.from(this.selectedStages);
        const btn = this.elements.reprocessBtn;

        btn.disabled = true;
        btn.classList.add('processing');
        btn.textContent = 'STARTING...';

        try {
            const config = {
                stages: stages,
                skip_existing: false,
                stage_options: {},
            };

            stages.forEach(stage => {
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

            await apiService.startProcessing(this.selectedProjectId, config);

            btn.textContent = 'PROCESSING...';

            stateManager.setState({
                currentProject: this.selectedProjectId,
                selectedStages: stages,
                processingActive: true,
            });

            this.startProcessingStatusPoll();

        } catch (error) {
            console.error('Failed to start processing:', error);
            btn.classList.remove('processing');
            btn.textContent = 'FAILED';
            btn.disabled = false;

            setTimeout(() => {
                this.updateProcessButton();
            }, 3000);
        }
    }

    startProcessingStatusPoll() {
        if (this.processingPollInterval) {
            clearInterval(this.processingPollInterval);
        }

        this.processingPollInterval = setInterval(async () => {
            await this.checkProcessingStatus();
        }, 2000);
    }

    stopProcessingStatusPoll() {
        if (this.processingPollInterval) {
            clearInterval(this.processingPollInterval);
            this.processingPollInterval = null;
        }
    }

    async checkProcessingStatus() {
        if (!this.selectedProjectId) {
            this.stopProcessingStatusPoll();
            return;
        }

        try {
            const jobData = await apiService.getJobStatus(this.selectedProjectId);
            const status = jobData.status;
            const btn = this.elements.reprocessBtn;

            if (status === 'running') {
                const stage = jobData.current_stage || 'processing';
                const lastOutput = jobData.last_output || '';

                if (stage === 'interactive') {
                    this.showInteractiveCompleteButton();
                } else {
                    this.hideInteractiveCompleteButton();
                }

                if (lastOutput && lastOutput.length > 0) {
                    btn.textContent = lastOutput;
                    btn.title = lastOutput;
                    btn.classList.add('showing-output');
                } else {
                    btn.textContent = `${stage.toUpperCase()}...`;
                    btn.classList.remove('showing-output');
                }
            } else if (status === 'completed' || status === 'complete') {
                this.stopProcessingStatusPoll();
                this.onProcessingComplete('complete');
            } else if (status === 'failed' || status === 'cancelled') {
                this.stopProcessingStatusPoll();
                this.onProcessingComplete('failed');
            } else if (status === 'idle') {
                const projectData = await apiService.getProject(this.selectedProjectId);
                if (projectData.status === 'complete' || projectData.status === 'failed') {
                    this.stopProcessingStatusPoll();
                    this.onProcessingComplete(projectData.status);
                }
            }
        } catch (error) {
            console.error('Failed to check processing status:', error);
        }
    }

    onProcessingComplete(status) {
        const btn = this.elements.reprocessBtn;
        btn.classList.remove('processing');

        if (status === 'complete') {
            btn.textContent = 'COMPLETE!';
            btn.classList.add('completed');
        } else {
            btn.textContent = 'FAILED';
            btn.classList.add('failed');
        }

        setTimeout(() => {
            btn.classList.remove('completed', 'failed');
            this.selectedStages.clear();
            this.updateProcessButton();
            this.showProjectDetails(this.selectedProjectId);
        }, 2000);

        this.loadProjects();

        stateManager.setState({
            processingActive: false,
        });

        this.hideInteractiveCompleteButton();
    }

    async completeInteractive() {
        if (!this.selectedProjectId) return;

        const btn = this.elements.interactiveCompleteBtn;
        if (!btn) return;

        btn.disabled = true;
        btn.textContent = 'SIGNALING...';

        try {
            await apiService.completeInteractive(this.selectedProjectId);
            btn.textContent = 'SIGNAL SENT';
            setTimeout(() => {
                this.hideInteractiveCompleteButton();
            }, 1000);
        } catch (error) {
            console.error('Failed to signal interactive complete:', error);
            btn.textContent = 'FAILED - TRY AGAIN';
            btn.disabled = false;
        }
    }

    showInteractiveCompleteButton() {
        const btn = this.elements.interactiveCompleteBtn;
        if (btn) {
            btn.classList.remove(CSS_CLASSES.HIDDEN);
            btn.disabled = false;
            btn.textContent = 'COMPLETE INTERACTIVE ROTO';
        }
    }

    hideInteractiveCompleteButton() {
        const btn = this.elements.interactiveCompleteBtn;
        if (btn) {
            btn.classList.add(CSS_CLASSES.HIDDEN);
            btn.disabled = false;
            btn.textContent = 'COMPLETE INTERACTIVE ROTO';
        }
    }

    hideDetailPanel() {
        if (this.elements.detailPanel) {
            dom.addClass(this.elements.detailPanel, CSS_CLASSES.HIDDEN);
        }
        if (this.elements.videoInfoSection) {
            dom.addClass(this.elements.videoInfoSection, CSS_CLASSES.HIDDEN);
        }
    }

    startAutoRefresh() {
        this.refreshInterval = setInterval(() => {
            this.loadProjects();
        }, 10000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();
        this.stopProcessingStatusPoll();

        if (this._boundHandlers.onUploadComplete) {
            stateManager.removeEventListener(EVENTS.UPLOAD_COMPLETE, this._boundHandlers.onUploadComplete);
        }
        this._boundHandlers = {};
    }
}
