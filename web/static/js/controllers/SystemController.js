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
import { ELEMENTS, EVENTS, CSS_CLASSES, UI } from '../config/constants.js';

export class SystemController {
    constructor() {
        this.elements = {
            systemStatus: dom.getElement(ELEMENTS.SYSTEM_STATUS),
            statusText: null,
            errorToast: dom.getElement(ELEMENTS.ERROR_TOAST),
            errorMessage: dom.getElement('error-message'),
            errorClose: dom.getElement('error-close'),
            shutdownBtn: dom.getElement('shutdown-btn'),
            sysOs: dom.getElement('sys-os'),
            sysDisk: dom.getElement('sys-disk'),
            sysProjects: dom.getElement('sys-projects'),
            sysGpu: dom.getElement('sys-gpu'),
        };

        if (this.elements.systemStatus) {
            this.elements.statusText = this.elements.systemStatus.querySelector('.status-text');
        }

        this.statusInterval = null;
        this._boundHandlers = {};
        this.setupEventListeners();
        this.checkSystemStatus();
        this.startAutoRefresh();
    }

    setupEventListeners() {
        this._boundHandlers.onErrorClose = () => this.hideError();
        this._boundHandlers.onShutdown = () => this.handleShutdown();
        this._boundHandlers.onError = (e) => this.showError(e.detail.message);
        this._boundHandlers.onStateChange = (e) => {
            if ('errorMessage' in e.detail.updates) {
                if (e.detail.newState.errorMessage) {
                    this.showError(e.detail.newState.errorMessage);
                } else {
                    this.hideError();
                }
            }
        };

        if (this.elements.errorClose) {
            this.elements.errorClose.addEventListener('click', this._boundHandlers.onErrorClose);
        }

        if (this.elements.shutdownBtn) {
            this.elements.shutdownBtn.addEventListener('click', this._boundHandlers.onShutdown);
        }

        stateManager.addEventListener(EVENTS.ERROR, this._boundHandlers.onError);
        stateManager.addEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);
    }

    async checkSystemStatus() {
        try {
            const status = await apiService.getSystemStatus();

            stateManager.setState({
                comfyuiOnline: status.comfyui,
                diskSpaceGB: status.disk_free_gb,
            });

            this.updateStatusDisplay(status.comfyui);
            this.updateSystemInfo(status);
        } catch (error) {
            console.error('Failed to check system status:', error);
            this.updateStatusDisplay(false);
        }
    }

    updateSystemInfo(status) {
        if (this.elements.sysOs) {
            dom.setText(this.elements.sysOs, status.os || '--');
        }

        if (this.elements.sysDisk) {
            const used = status.disk_total_gb - status.disk_free_gb;
            const percent = status.disk_used_percent || 0;
            dom.setText(this.elements.sysDisk, `${percent}% (${used.toFixed(0)}GB / ${status.disk_total_gb}GB)`);
        }

        if (this.elements.sysProjects) {
            dom.setText(this.elements.sysProjects, `${status.projects_size_gb || 0} GB`);
        }

        if (this.elements.sysGpu) {
            const gpuName = status.gpu_name || 'Unknown';
            const vram = status.gpu_vram_gb || 0;
            dom.setText(this.elements.sysGpu, `${gpuName} ${vram}GB`);
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

        setTimeout(() => {
            this.hideError();
        }, UI.ERROR_TOAST_DURATION);
    }

    hideError() {
        if (this.elements.errorToast) {
            dom.hide(this.elements.errorToast);
        }
        stateManager.clearError();
    }

    async handleShutdown() {
        const confirmed = confirm(
            'Shutdown the VFX Pipeline platform?\n\n' +
            'The web interface will close and the server will stop.\n' +
            'You can restart it by running: ./start-platform.sh'
        );

        if (!confirmed) {
            return;
        }

        try {
            // Disable the button to prevent double-clicks
            if (this.elements.shutdownBtn) {
                this.elements.shutdownBtn.disabled = true;
                dom.setText(this.elements.shutdownBtn, 'Shutting down...');
            }

            // Call shutdown API
            await apiService.shutdownSystem();

            // Show confirmation message
            this.showError('Server is shutting down. You can close this window.');

            // Stop auto-refresh
            this.stopAutoRefresh();

            setTimeout(() => {
                document.body.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center; height: 100vh; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                        <div style="text-align: center;">
                            <h1 style="color: #2ecc71; margin-bottom: 20px;">✓ Shutdown Complete</h1>
                            <p style="font-size: 1.2em; margin-bottom: 30px;">The VFX Pipeline platform has been shut down successfully.</p>
                            <p style="color: #888;">You can close this window.</p>
                            <p style="color: #888; margin-top: 20px;">To restart: <code style="background: #f5f5f5; padding: 4px 8px; border-radius: 4px;">./start-platform.sh</code></p>
                        </div>
                    </div>
                `;
            }, UI.SHUTDOWN_MESSAGE_DELAY);

        } catch (error) {
            console.error('Failed to shutdown system:', error);
            this.showError('Failed to shutdown server. Please try again or use ./stop-platform.sh');

            // Re-enable button on error
            if (this.elements.shutdownBtn) {
                this.elements.shutdownBtn.disabled = false;
                dom.setText(this.elements.shutdownBtn, 'Shutdown');
            }
        }
    }

    startAutoRefresh() {
        this.statusInterval = setInterval(() => {
            this.checkSystemStatus();
        }, UI.SYSTEM_STATUS_INTERVAL);
    }

    stopAutoRefresh() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();

        if (this.elements.errorClose && this._boundHandlers.onErrorClose) {
            this.elements.errorClose.removeEventListener('click', this._boundHandlers.onErrorClose);
        }
        if (this.elements.shutdownBtn && this._boundHandlers.onShutdown) {
            this.elements.shutdownBtn.removeEventListener('click', this._boundHandlers.onShutdown);
        }

        stateManager.removeEventListener(EVENTS.ERROR, this._boundHandlers.onError);
        stateManager.removeEventListener(EVENTS.STATE_CHANGED, this._boundHandlers.onStateChange);

        this._boundHandlers = {};
    }
}
