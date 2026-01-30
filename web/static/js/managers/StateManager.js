/**
 * StateManager - Centralized state management
 *
 * Provides a single source of truth for application state with:
 * - Immutable state updates
 * - Event-driven state changes
 * - State validation
 * - No direct DOM manipulation
 *
 * Follows the Single Responsibility Principle by only managing state.
 */

import { EVENTS } from '../config/constants.js';

export class StateManager extends EventTarget {
    constructor() {
        super();
        this._state = {
            // Application state
            config: null,
            currentView: 'upload',

            // Project state
            projectId: null,
            projectDir: null,
            projectName: null,
            videoInfo: null,
            vramAnalysis: null,
            selectedStages: [],
            rotoPrompt: 'person',
            skipExisting: false,

            // Processing state
            isProcessing: false,
            progress: 0,
            currentStage: null,
            currentStageIndex: 0,
            totalStages: 0,
            frame: 0,
            totalFrames: 0,
            startTime: null,
            elapsedTime: 0,
            estimatedTimeRemaining: null,

            // WebSocket state
            wsConnected: false,
            wsReconnectAttempts: 0,

            // System state
            comfyuiOnline: false,
            diskSpaceGB: 0,

            // Projects list
            projects: [],

            // UI state
            uploadProgress: 0,
            uploadFilename: null,
            errorMessage: null,
        };
    }

    /**
     * Get the current state (read-only)
     */
    get state() {
        // Return a frozen copy to prevent direct mutation
        return Object.freeze({ ...this._state });
    }

    /**
     * Update state with partial update
     * @param {Object} updates - Partial state updates
     */
    setState(updates) {
        const oldState = { ...this._state };
        this._state = {
            ...this._state,
            ...updates,
        };

        // Emit state changed event
        this.dispatchEvent(new CustomEvent(EVENTS.STATE_CHANGED, {
            detail: {
                oldState,
                newState: this.state,
                updates,
            },
        }));
    }

    /**
     * Reset state to initial values
     */
    reset() {
        this.setState({
            projectId: null,
            projectDir: null,
            projectName: null,
            videoInfo: null,
            vramAnalysis: null,
            selectedStages: [],
            isProcessing: false,
            progress: 0,
            currentStage: null,
            currentStageIndex: 0,
            totalStages: 0,
            frame: 0,
            totalFrames: 0,
            startTime: null,
            elapsedTime: 0,
            estimatedTimeRemaining: null,
            uploadProgress: 0,
            uploadFilename: null,
            errorMessage: null,
        });
    }

    /**
     * Get a specific state value
     * @param {string} key - State key
     * @returns {*} State value
     */
    get(key) {
        return this._state[key];
    }

    /**
     * Check if processing is active
     * @returns {boolean}
     */
    isActive() {
        return this._state.isProcessing && this._state.projectId !== null;
    }

    /**
     * Update processing progress
     * @param {Object} data - Progress data from WebSocket
     */
    updateProgress(data) {
        const updates = {};

        if (data.progress !== undefined) {
            updates.progress = data.progress;
        }

        if (data.stage) {
            updates.currentStage = data.stage;
            updates.currentStageIndex = data.stage_index ?? this._state.currentStageIndex;
        }

        if (data.total_stages !== undefined) {
            updates.totalStages = data.total_stages;
        }

        if (data.frame !== undefined) {
            updates.frame = data.frame;
        }

        if (data.total_frames !== undefined) {
            updates.totalFrames = data.total_frames;
        }

        if (data.message) {
            // Don't store in state, just emit event
            this.dispatchEvent(new CustomEvent('logMessage', {
                detail: { message: data.message },
            }));
        }

        if (Object.keys(updates).length > 0) {
            this.setState(updates);

            // Emit progress update event
            this.dispatchEvent(new CustomEvent(EVENTS.PROGRESS_UPDATE, {
                detail: { ...updates },
            }));
        }
    }

    /**
     * Start processing
     * @param {string} projectId - Project ID
     * @param {string} projectName - Project name
     * @param {Array<string>} stages - Selected stages
     */
    startProcessing(projectId, projectName, stages) {
        this.setState({
            projectId,
            projectName,
            selectedStages: stages,
            isProcessing: true,
            progress: 0,
            currentStage: stages[0] || null,
            currentStageIndex: 0,
            totalStages: stages.length,
            frame: 0,
            totalFrames: 0,
            startTime: Date.now(),
        });
    }

    /**
     * Stop processing
     */
    stopProcessing() {
        this.setState({
            isProcessing: false,
            startTime: null,
        });
    }

    /**
     * Set configuration (loaded from API)
     * @param {Object} config - Configuration object
     */
    setConfig(config) {
        this.setState({ config });
        this.dispatchEvent(new CustomEvent(EVENTS.CONFIG_LOADED, {
            detail: { config },
        }));
    }

    /**
     * Show error
     * @param {string} message - Error message
     */
    showError(message) {
        this.setState({ errorMessage: message });
        this.dispatchEvent(new CustomEvent(EVENTS.ERROR, {
            detail: { message },
        }));
    }

    /**
     * Clear error
     */
    clearError() {
        this.setState({ errorMessage: null });
    }
}

// Export singleton instance
export const stateManager = new StateManager();
