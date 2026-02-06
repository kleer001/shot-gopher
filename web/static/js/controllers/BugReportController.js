/**
 * BugReportController - Manages bug report button and dialog
 *
 * Responsibilities:
 * - Show/hide bug report dialog
 * - Collect system info from API
 * - Format and copy report to clipboard
 * - Open mailto link for email submission
 *
 * Follows Single Responsibility Principle by only handling bug report UI.
 */

import { apiService } from '../services/APIService.js';
import { ELEMENTS, CSS_CLASSES, UI } from '../config/constants.js';
import * as dom from '../utils/dom.js';

const MAINTAINER_EMAIL = 'kleer001code@gmail.com';

export class BugReportController {
    constructor() {
        this.elements = {
            btn: dom.getElement(ELEMENTS.BUG_REPORT_BTN),
            dialog: dom.getElement(ELEMENTS.BUG_REPORT_DIALOG),
            description: dom.getElement(ELEMENTS.BUG_DESCRIPTION),
            expected: dom.getElement(ELEMENTS.BUG_EXPECTED),
            copyBtn: dom.getElement(ELEMENTS.BUG_REPORT_COPY),
            emailBtn: dom.getElement(ELEMENTS.BUG_REPORT_EMAIL),
            cancelBtn: dom.getElement(ELEMENTS.BUG_REPORT_CANCEL),
            systemInfo: dom.getElement(ELEMENTS.BUG_SYSTEM_INFO),
        };

        this.systemStatus = null;
        this._boundHandlers = {};
        this._setupEventListeners();
    }

    _setupEventListeners() {
        this._boundHandlers.onOpen = () => this.open();
        this._boundHandlers.onCopy = () => this._copyReport();
        this._boundHandlers.onEmail = () => this._emailReport();
        this._boundHandlers.onCancel = () => this.close();
        this._boundHandlers.onOverlayClick = (e) => {
            if (e.target === this.elements.dialog) {
                this.close();
            }
        };

        this.elements.btn.addEventListener('click', this._boundHandlers.onOpen);
        this.elements.copyBtn.addEventListener('click', this._boundHandlers.onCopy);
        this.elements.emailBtn.addEventListener('click', this._boundHandlers.onEmail);
        this.elements.cancelBtn.addEventListener('click', this._boundHandlers.onCancel);
        this.elements.dialog.addEventListener('click', this._boundHandlers.onOverlayClick);
    }

    async open() {
        dom.show(this.elements.dialog);
        this.elements.description.value = '';
        this.elements.expected.value = '';
        this.elements.description.focus();
        await this._loadSystemInfo();
    }

    close() {
        dom.hide(this.elements.dialog);
    }

    async _loadSystemInfo() {
        dom.setText(this.elements.systemInfo, 'Loading system info...');
        try {
            this.systemStatus = await apiService.getSystemStatus();
            const parts = [];
            if (this.systemStatus.os) parts.push(`OS: ${this.systemStatus.os}`);
            if (this.systemStatus.gpu_name) parts.push(`GPU: ${this.systemStatus.gpu_name}`);
            if (this.systemStatus.gpu_vram_gb) parts.push(`VRAM: ${this.systemStatus.gpu_vram_gb}GB`);
            dom.setText(
                this.elements.systemInfo,
                parts.length > 0 ? parts.join(' | ') : 'System info loaded'
            );
        } catch {
            dom.setText(this.elements.systemInfo, 'Could not load system info');
            this.systemStatus = null;
        }
    }

    _buildReport() {
        const description = this.elements.description.value.trim();
        const expected = this.elements.expected.value.trim();

        let report = '[Bug Report] Shot Gopher\n\n';

        if (description) {
            report += `What happened:\n${description}\n\n`;
        }
        if (expected) {
            report += `Expected behavior:\n${expected}\n\n`;
        }

        report += 'System Information:\n';
        if (this.systemStatus) {
            if (this.systemStatus.os) report += `  OS: ${this.systemStatus.os}\n`;
            if (this.systemStatus.gpu_name) report += `  GPU: ${this.systemStatus.gpu_name}\n`;
            if (this.systemStatus.gpu_vram_gb) report += `  VRAM: ${this.systemStatus.gpu_vram_gb} GB\n`;
            report += `  ComfyUI: ${this.systemStatus.comfyui ? 'Online' : 'Offline'}\n`;
            if (this.systemStatus.disk_free_gb != null) {
                report += `  Disk Free: ${this.systemStatus.disk_free_gb} GB\n`;
            }
        } else {
            report += '  (unavailable)\n';
        }

        report += `\nTimestamp: ${new Date().toISOString()}\n`;

        return report;
    }

    async _copyReport() {
        const report = this._buildReport();
        await navigator.clipboard.writeText(report);
        dom.setText(this.elements.copyBtn, 'Copied!');
        setTimeout(() => {
            dom.setText(this.elements.copyBtn, 'Copy Report');
        }, UI.BUTTON_RESET_DELAY);
    }

    _emailReport() {
        const report = this._buildReport();
        const subject = encodeURIComponent('[Bug Report] Shot Gopher');
        const body = encodeURIComponent(report);
        window.open(`mailto:${MAINTAINER_EMAIL}?subject=${subject}&body=${body}`);
    }

    destroy() {
        this.elements.btn.removeEventListener('click', this._boundHandlers.onOpen);
        this.elements.copyBtn.removeEventListener('click', this._boundHandlers.onCopy);
        this.elements.emailBtn.removeEventListener('click', this._boundHandlers.onEmail);
        this.elements.cancelBtn.removeEventListener('click', this._boundHandlers.onCancel);
        this.elements.dialog.removeEventListener('click', this._boundHandlers.onOverlayClick);
        this._boundHandlers = {};
    }
}
