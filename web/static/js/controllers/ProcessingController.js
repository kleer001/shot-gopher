/**
 * ProcessingController - Manages processing panel UI
 *
 * Responsibilities:
 * - Display processing progress
 * - Update progress bars/rings
 * - Show stage status
 * - Display logs
 * - Handle WebSocket messages
 * - Cancel processing
 *
 * Follows Single Responsibility Principle by only handling processing UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { wsService } from '../services/WebSocketService.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { formatTime, getElapsedTime } from '../utils/time.js';
import { ELEMENTS, EVENTS } from '../config/constants.js';

export class ProcessingController {
    constructor() {
        this.elements = {
            processingPanel: dom.getElement(ELEMENTS.PROCESSING_PANEL),
            processingProjectName: dom.getElement('processing-project-name'),
            cancelBtn: dom.getElement('cancel-processing-btn'),
            progressPercent: dom.getElement(ELEMENTS.PROGRESS_PERCENT),
            currentStageName: dom.getElement(ELEMENTS.CURRENT_STAGE_NAME),
            currentStageLabel: dom.getElement('current-stage-label'),
            progressFrames: dom.getElement('progress-frames'),
            elapsedTime: dom.getElement('elapsed-time'),
            remainingTime: dom.getElement('remaining-time'),
            stagesListProgress: dom.getElement('stages-list-progress'),
            logOutput: dom.getElement(ELEMENTS.LOG_OUTPUT),
            clearLogsBtn: dom.getElement('clear-logs-btn'),
            processingProgressFill: dom.getElement('processing-progress-fill'),
            progressRing: dom.getElement('progress-ring'),
            progressCircle: dom.getElement('progress-circle'),
        };

        this.timerInterval = null;
        this._boundHandlers = {};
        this.setupEventListeners();
    }

    setupEventListeners() {
        this._boundHandlers.onCancelClick = () => this.handleCancel();
        this._boundHandlers.onClearLogsClick = () => this.clearLogs();
        this._boundHandlers.onStateChange = (e) => this.handleStateChange(e.detail);
        this._boundHandlers.onProgressUpdate = (e) => this.handleProgressUpdate(e.detail);
        this._boundHandlers.onLogMessage = (e) => this.appendLog(e.detail.message);
        this._boundHandlers.onWsMessage = (e) => stateManager.updateProgress(e.detail);
        this._boundHandlers.onWsConnected = () => console.log('WebSocket connected');
        this._boundHandlers.onWsDisconnected = () => console.log('WebSocket disconnected');

        if (this.elements.cancelBtn) {
            this.elements.cancelBtn.addEventListener('click', this._boundHandlers.onCancelClick);
        }

        if (this.elements.clearLogsBtn) {
            this.elements.clearLogsBtn.addEventListener('click', this._boundHandlers.onClearLogsClick);
        }

        stateManager.addEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);
        stateManager.addEventListener(EVENTS.PROGRESS_UPDATE, this._boundHandlers.onProgressUpdate);
        stateManager.addEventListener('logMessage', this._boundHandlers.onLogMessage);

        wsService.addEventListener('message', this._boundHandlers.onWsMessage);
        wsService.addEventListener('connected', this._boundHandlers.onWsConnected);
        wsService.addEventListener('disconnected', this._boundHandlers.onWsDisconnected);
    }

    handleStateChange(detail) {
        const { newState, updates } = detail;

        // Show/hide processing panel based on isProcessing
        if ('isProcessing' in updates) {
            if (newState.isProcessing) {
                this.showProcessingPanel();
                this.setupProcessingUI();
                this.startTimer();
                // Connect WebSocket
                if (newState.projectId) {
                    wsService.connect(newState.projectId);
                }
            } else {
                this.stopTimer();
                wsService.disconnect();
            }
        }
    }

    showProcessingPanel() {
        dom.show(this.elements.processingPanel);
    }

    hideProcessingPanel() {
        dom.hide(this.elements.processingPanel);
    }

    setupProcessingUI() {
        const state = stateManager.state;
        const config = state.config;

        // Set project name
        dom.setText(this.elements.processingProjectName, state.projectName || '--');

        // Build stages list (escape values for defense in depth)
        const stagesHTML = state.selectedStages.map(stageId => {
            const stageName = config?.stages?.[stageId]?.name || stageId;
            const safeStageId = dom.escapeHTML(stageId);
            const safeStageName = dom.escapeHTML(stageName);
            return `
                <div class="stage-item pending" data-stage="${safeStageId}">
                    <span class="stage-icon">○</span>
                    <span>${safeStageName}</span>
                    <span class="stage-status">pending</span>
                </div>
            `;
        }).join('');

        dom.setHTML(this.elements.stagesListProgress, stagesHTML);

        // Reset progress displays
        this.updateProgress(0);
        dom.setText(this.elements.progressPercent, '0%');
        dom.setText(this.elements.progressFrames, 'Frame 0 / 0');
        dom.setText(this.elements.currentStageLabel, `STAGE 1/${state.totalStages}`);

        const firstStageName = config?.stages?.[state.selectedStages[0]]?.name || state.selectedStages[0] || 'IDLE';
        dom.setText(this.elements.currentStageName, firstStageName.toUpperCase());

        // Clear logs
        this.clearLogs();
    }

    handleProgressUpdate(data) {
        const state = stateManager.state;
        const config = state.config;

        // Update progress bar/ring
        if (state.progress > 0) {
            const percent = Math.round(state.progress * 100);
            this.updateProgress(percent);
            dom.setText(this.elements.progressPercent, `${percent}%`);
        }

        // Update frame count
        if (state.frame > 0 && state.totalFrames > 0) {
            dom.setText(this.elements.progressFrames, `Frame ${state.frame} / ${state.totalFrames}`);
        }

        // Update current stage
        if (state.currentStage) {
            const stageIndex = state.currentStageIndex;
            const stageName = config?.stages?.[state.currentStage]?.name || state.currentStage;

            dom.setText(this.elements.currentStageLabel, `STAGE ${stageIndex + 1}/${state.totalStages}`);
            dom.setText(this.elements.currentStageName, stageName.toUpperCase());

            // Update stages list
            this.updateStagesList(state.currentStage, stageIndex);
        }
    }

    updateProgress(percent) {
        // Update linear progress bar
        if (this.elements.processingProgressFill) {
            this.elements.processingProgressFill.style.width = `${percent}%`;
        }

        // Update SVG progress ring (Dashboard template)
        if (this.elements.progressRing) {
            const radius = 54;
            const circumference = 2 * Math.PI * radius;
            const offset = circumference * (1 - percent / 100);
            this.elements.progressRing.style.strokeDasharray = circumference;
            this.elements.progressRing.style.strokeDashoffset = offset;
        }

        // Update SVG progress circle (Split template)
        if (this.elements.progressCircle) {
            const radius = 90;
            const circumference = 2 * Math.PI * radius;
            const offset = circumference * (1 - percent / 100);
            this.elements.progressCircle.style.strokeDasharray = circumference;
            this.elements.progressCircle.style.strokeDashoffset = offset;
        }
    }

    updateStagesList(currentStage, currentIndex) {
        const stageItems = this.elements.stagesListProgress?.querySelectorAll('.stage-item');
        if (!stageItems) return;

        stageItems.forEach((item, index) => {
            const icon = item.querySelector('.stage-icon');
            const status = item.querySelector('.stage-status');

            if (index < currentIndex) {
                // Completed
                icon.className = 'stage-icon completed';
                icon.innerHTML = '✓';
                if (status) status.textContent = 'completed';
                dom.addClass(item, 'completed');
                dom.removeClass(item, 'processing');
                dom.removeClass(item, 'pending');
            } else if (index === currentIndex) {
                // Processing
                icon.className = 'stage-icon processing';
                icon.innerHTML = '⟳';
                if (status) status.textContent = 'processing...';
                dom.addClass(item, 'processing');
                dom.removeClass(item, 'completed');
                dom.removeClass(item, 'pending');
            } else {
                // Pending
                icon.className = 'stage-icon pending';
                icon.innerHTML = '○';
                if (status) status.textContent = 'pending';
                dom.addClass(item, 'pending');
                dom.removeClass(item, 'completed');
                dom.removeClass(item, 'processing');
            }
        });
    }

    appendLog(message) {
        if (!this.elements.logOutput) return;

        this.elements.logOutput.textContent += message + '\n';
        this.elements.logOutput.scrollTop = this.elements.logOutput.scrollHeight;
    }

    clearLogs() {
        if (this.elements.logOutput) {
            this.elements.logOutput.textContent = '';
        }
    }

    startTimer() {
        this.stopTimer();

        this.timerInterval = setInterval(() => {
            this.updateTimer();
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    updateTimer() {
        const state = stateManager.state;
        if (!state.startTime) return;

        const elapsed = getElapsedTime(state.startTime);
        dom.setText(this.elements.elapsedTime, formatTime(elapsed));

        // Update estimated remaining time
        if (state.progress > 0 && state.progress < 1) {
            const totalEstimate = elapsed / state.progress;
            const remaining = Math.round(totalEstimate - elapsed);
            dom.setText(this.elements.remainingTime, formatTime(remaining));
        }
    }

    async handleCancel() {
        const projectId = stateManager.get('projectId');
        if (!projectId) return;

        if (!confirm('Are you sure you want to cancel processing?')) {
            return;
        }

        try {
            await apiService.cancelProcessing(projectId);
            stateManager.stopProcessing();
            this.hideProcessingPanel();
        } catch (error) {
            stateManager.showError(error.message || 'Failed to cancel processing');
        }
    }

    reset() {
        this.stopTimer();
        this.clearLogs();
        this.hideProcessingPanel();
    }

    destroy() {
        this.stopTimer();

        if (this.elements.cancelBtn && this._boundHandlers.onCancelClick) {
            this.elements.cancelBtn.removeEventListener('click', this._boundHandlers.onCancelClick);
        }
        if (this.elements.clearLogsBtn && this._boundHandlers.onClearLogsClick) {
            this.elements.clearLogsBtn.removeEventListener('click', this._boundHandlers.onClearLogsClick);
        }

        stateManager.removeEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);
        stateManager.removeEventListener(EVENTS.PROGRESS_UPDATE, this._boundHandlers.onProgressUpdate);
        stateManager.removeEventListener('logMessage', this._boundHandlers.onLogMessage);

        wsService.removeEventListener('message', this._boundHandlers.onWsMessage);
        wsService.removeEventListener('connected', this._boundHandlers.onWsConnected);
        wsService.removeEventListener('disconnected', this._boundHandlers.onWsDisconnected);

        this._boundHandlers = {};
    }
}
