/**
 * SystemController - Manages system status display
 *
 * Responsibilities:
 * - Check ComfyUI status
 * - Display online/offline status
 * - Auto-refresh status
 * - Handle error toast display
 *
 * Follows Single Responsibility Principle by only handling system status UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { ELEMENTS, EVENTS, CSS_CLASSES } from '../config/constants.js';

export class SystemController {
    constructor() {
        this.elements = {
            systemStatus: dom.getElement(ELEMENTS.SYSTEM_STATUS),
            statusText: null, // Will be found inside systemStatus
            errorToast: dom.getElement(ELEMENTS.ERROR_TOAST),
            errorMessage: dom.getElement('error-message'),
            errorClose: dom.getElement('error-close'),
        };

        // Find status text element
        if (this.elements.systemStatus) {
            this.elements.statusText = this.elements.systemStatus.querySelector('.status-text');
        }

        this.statusInterval = null;
        this.setupEventListeners();
        this.checkSystemStatus();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        // Error toast close button
        if (this.elements.errorClose) {
            this.elements.errorClose.addEventListener('click', () => {
                this.hideError();
            });
        }

        // Listen to error events from state
        stateManager.addEventListener(EVENTS.ERROR, (e) => {
            this.showError(e.detail.message);
        });

        // Listen to state changes
        stateManager.addEventListener(EVENTS.STATE_CHANGED, (e) => {
            if ('errorMessage' in e.detail.updates) {
                if (e.detail.newState.errorMessage) {
                    this.showError(e.detail.newState.errorMessage);
                } else {
                    this.hideError();
                }
            }
        });
    }

    async checkSystemStatus() {
        try {
            const status = await apiService.getSystemStatus();

            // Update state
            stateManager.setState({
                comfyuiOnline: status.comfyui,
                diskSpaceGB: status.disk_space_gb,
            });

            // Update UI
            this.updateStatusDisplay(status.comfyui);
        } catch (error) {
            console.error('Failed to check system status:', error);
            this.updateStatusDisplay(false);
        }
    }

    updateStatusDisplay(isOnline) {
        if (!this.elements.systemStatus) return;

        if (isOnline) {
            dom.addClass(this.elements.systemStatus, CSS_CLASSES.ONLINE);
            dom.removeClass(this.elements.systemStatus, CSS_CLASSES.OFFLINE);
            if (this.elements.statusText) {
                dom.setText(this.elements.statusText, 'ComfyUI Online');
            }
        } else {
            dom.removeClass(this.elements.systemStatus, CSS_CLASSES.ONLINE);
            dom.addClass(this.elements.systemStatus, CSS_CLASSES.OFFLINE);
            if (this.elements.statusText) {
                dom.setText(this.elements.statusText, 'ComfyUI Offline');
            }
        }
    }

    showError(message) {
        if (!this.elements.errorToast || !this.elements.errorMessage) return;

        dom.setText(this.elements.errorMessage, message);
        dom.show(this.elements.errorToast);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        if (this.elements.errorToast) {
            dom.hide(this.elements.errorToast);
        }
        stateManager.clearError();
    }

    startAutoRefresh() {
        // Check system status every 30 seconds
        this.statusInterval = setInterval(() => {
            this.checkSystemStatus();
        }, 30000);
    }

    stopAutoRefresh() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();
    }
}
