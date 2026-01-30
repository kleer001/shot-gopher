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
import { ELEMENTS, CSS_CLASSES } from '../config/constants.js';

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
    return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
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
            detailVideoSection: dom.getElement(ELEMENTS.DETAIL_VIDEO_SECTION),
            detailVideoInfo: dom.getElement(ELEMENTS.DETAIL_VIDEO_INFO),
            openFolderBtn: dom.getElement(ELEMENTS.OPEN_FOLDER_BTN),
            deleteProjectBtn: dom.getElement(ELEMENTS.DELETE_PROJECT_BTN),
            reprocessBtn: dom.getElement(ELEMENTS.REPROCESS_BTN),
        };

        this.selectedProjectId = null;
        this.selectedStages = new Set();
        this.refreshInterval = null;
        this.processingPollInterval = null;

        this.bindDetailActions();
        this.loadProjects();
        this.startAutoRefresh();
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

        await this.showProjectDetails(projectId);
    }

    async showProjectDetails(projectId) {
        if (!this.elements.detailPanel) return;

        dom.removeClass(this.elements.detailPanel, CSS_CLASSES.HIDDEN);

        if (this.elements.detailProjectName) {
            this.elements.detailProjectName.textContent = projectId;
        }

        try {
            const [projectData, outputsData, vramData] = await Promise.all([
                apiService.getProject(projectId).catch(() => null),
                apiService.getProjectOutputs(projectId).catch(() => null),
                apiService.getProjectVram(projectId).catch(() => null),
            ]);

            this.renderStagesStatus(projectData, outputsData, vramData);
            this.renderVramInfo(vramData);
        } catch (error) {
            console.error('Failed to load project details:', error);
        }
    }

    renderStagesStatus(projectData, outputsData, vramData) {
        if (!this.elements.detailStages) return;

        const outputs = outputsData?.outputs || {};
        const vramStages = vramData?.analysis?.stages || {};

        const stageLabels = {
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

            let filesInfo = '';
            if (isCompleted && fileCount > 0) {
                const totalBytes = (outputData?.files || []).reduce((sum, f) => sum + (f.size || 0), 0);
                const sizeStr = formatFileSize(totalBytes);
                filesInfo = `${fileCount} Files / ${sizeStr}`;
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

            return `
            <div class="stage-status-item selectable ${stageClass}" data-stage="${stage}">
                <div class="stage-marker"></div>
                <span class="stage-status-name">${label}</span>
                <span class="stage-vram ${vramStatusClass}" title="${dom.escapeHTML(vramTitle)}">${vramDisplay}</span>
                <span class="stage-status-info">${filesInfo}</span>
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

        if (this.selectedStages.has(stage)) {
            this.selectedStages.delete(stage);
            item.classList.remove('selected');
        } else {
            this.selectedStages.add(stage);
            item.classList.add('selected');
        }

        this.updateProcessButton();
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

    renderVramInfo(vramData) {
        if (!this.elements.vramInfoSection || !this.elements.vramInfo) return;

        const analysis = vramData?.analysis;
        if (!analysis) {
            dom.addClass(this.elements.vramInfoSection, CSS_CLASSES.HIDDEN);
            return;
        }

        dom.removeClass(this.elements.vramInfoSection, CSS_CLASSES.HIDDEN);

        const requiredGb = (analysis.peak_vram_mb || 0) / 1024;
        const availableGb = (analysis.available_vram_mb || 0) / 1024;
        const hasWarning = analysis.warning || requiredGb > availableGb;
        const valueClass = hasWarning ? (requiredGb > availableGb ? 'danger' : 'warning') : 'ok';

        let html = `
            <div class="vram-row">
                <span class="vram-label">Required</span>
                <span class="vram-value ${valueClass}">${requiredGb.toFixed(1)} GB</span>
            </div>
            <div class="vram-row">
                <span class="vram-label">Available</span>
                <span class="vram-value">${availableGb.toFixed(1)} GB</span>
            </div>
        `;

        if (analysis.chunking_required) {
            html += `
            <div class="vram-note">
                Chunked processing required (${analysis.chunks || 'multiple'} chunks)
            </div>
        `;
        }

        if (analysis.warning && !analysis.chunking_required) {
            html += `
            <div class="vram-note">
                ${dom.escapeHTML(analysis.warning)}
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
                roto_prompt: 'person',
                skip_existing: false,
            };

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
            btn.textContent = 'FAILED - RETRY?';
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
            const projectData = await apiService.getProject(this.selectedProjectId);
            const status = projectData.status;

            if (status === 'complete' || status === 'failed') {
                this.stopProcessingStatusPoll();
                this.onProcessingComplete(status);
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
    }

    hideDetailPanel() {
        if (this.elements.detailPanel) {
            dom.addClass(this.elements.detailPanel, CSS_CLASSES.HIDDEN);
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
    }
}
