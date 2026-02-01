/**
 * UploadController - Manages file upload UI and logic
 *
 * Responsibilities:
 * - Handle drag-and-drop file uploads
 * - Manage upload progress display
 * - Validate file types
 * - Dispatch upload complete event for project selection
 *
 * Follows Single Responsibility Principle by only handling upload UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { ELEMENTS, EVENTS, UPLOAD } from '../config/constants.js';

export class UploadController {
    constructor() {
        this.elements = {
            dropZone: dom.getElement(ELEMENTS.DROP_ZONE),
            fileInput: dom.getElement(ELEMENTS.FILE_INPUT),
            browseBtn: dom.getElement(ELEMENTS.BROWSE_BTN),
            uploadProgress: dom.getElement(ELEMENTS.UPLOAD_PROGRESS),
            uploadFilename: dom.getElement('upload-filename'),
            uploadProgressFill: dom.getElement('upload-progress-fill'),
            uploadPercentText: dom.getElement('upload-percent-text'),
            overwriteDialog: dom.getElement(ELEMENTS.OVERWRITE_DIALOG),
            overwriteProjectName: dom.getElement(ELEMENTS.OVERWRITE_PROJECT_NAME),
            overwriteYes: dom.getElement(ELEMENTS.OVERWRITE_YES),
            overwriteNo: dom.getElement(ELEMENTS.OVERWRITE_NO),
            overwriteCancel: dom.getElement(ELEMENTS.OVERWRITE_CANCEL),
        };

        this._boundHandlers = {};
        this._pendingFile = null;
        this._pendingProjectName = null;
        this.setupEventListeners();
    }

    setupEventListeners() {
        this._boundHandlers.onBrowseClick = () => this.elements.fileInput?.click();
        this._boundHandlers.onFileChange = (e) => {
            const file = e.target.files?.[0];
            if (file) this.handleFileSelected(file);
        };
        this._boundHandlers.onDragOver = (e) => this.handleDragOver(e);
        this._boundHandlers.onDragLeave = (e) => this.handleDragLeave(e);
        this._boundHandlers.onDrop = (e) => this.handleDrop(e);

        if (this.elements.browseBtn) {
            this.elements.browseBtn.addEventListener('click', this._boundHandlers.onBrowseClick);
        }

        if (this.elements.fileInput) {
            this.elements.fileInput.addEventListener('change', this._boundHandlers.onFileChange);
        }

        if (this.elements.dropZone) {
            this.elements.dropZone.addEventListener('dragover', this._boundHandlers.onDragOver);
            this.elements.dropZone.addEventListener('dragleave', this._boundHandlers.onDragLeave);
            this.elements.dropZone.addEventListener('drop', this._boundHandlers.onDrop);
        }

        this._boundHandlers.onOverwriteYes = () => this.handleOverwriteChoice(true);
        this._boundHandlers.onOverwriteNo = () => this.handleOverwriteChoice(false);
        this._boundHandlers.onOverwriteCancel = () => this.handleOverwriteChoice(false);

        if (this.elements.overwriteYes) {
            this.elements.overwriteYes.addEventListener('click', this._boundHandlers.onOverwriteYes);
        }
        if (this.elements.overwriteNo) {
            this.elements.overwriteNo.addEventListener('click', this._boundHandlers.onOverwriteNo);
        }
        if (this.elements.overwriteCancel) {
            this.elements.overwriteCancel.addEventListener('click', this._boundHandlers.onOverwriteCancel);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        dom.addClass(this.elements.dropZone, 'drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        dom.removeClass(this.elements.dropZone, 'drag-over');
    }

    handleDrop(e) {
        e.preventDefault();
        dom.removeClass(this.elements.dropZone, 'drag-over');

        const file = e.dataTransfer?.files?.[0];
        if (file) {
            this.handleFileSelected(file);
        }
    }

    async handleFileSelected(file, overwrite = false) {
        const lastDotIndex = file.name.lastIndexOf('.');
        const ext = lastDotIndex > 0 ? file.name.slice(lastDotIndex).toLowerCase() : '';
        const config = stateManager.get('config');
        const supportedFormats = config?.supportedVideoFormats || UPLOAD.SUPPORTED_FORMATS;

        if (!ext || !supportedFormats.includes(ext)) {
            const displayExt = ext || '(no extension)';
            stateManager.showError(`Unsupported file type: ${displayExt}`);
            return;
        }

        dom.show(this.elements.uploadProgress);
        dom.setText(this.elements.uploadFilename, file.name);
        stateManager.setState({ uploadFilename: file.name, uploadProgress: 0 });

        try {
            const result = await apiService.uploadVideo(file, (progress) => {
                this.updateUploadProgress(progress);
            }, { overwrite });

            this.handleUploadSuccess(result);
        } catch (error) {
            if (error.status === 409) {
                this.showOverwriteDialog(file, error.message);
            } else {
                this.handleUploadError(error);
            }
        }
    }

    showOverwriteDialog(file, errorMessage) {
        dom.hide(this.elements.uploadProgress);
        this._pendingFile = file;

        const match = errorMessage.match(/Project '([^']+)' already exists/);
        this._pendingProjectName = match ? match[1] : file.name.replace(/\.[^/.]+$/, '');

        dom.setText(this.elements.overwriteProjectName, this._pendingProjectName);
        dom.show(this.elements.overwriteDialog);
    }

    hideOverwriteDialog() {
        dom.hide(this.elements.overwriteDialog);
        this._pendingFile = null;
        this._pendingProjectName = null;
    }

    handleOverwriteChoice(shouldOverwrite) {
        const file = this._pendingFile;
        this.hideOverwriteDialog();

        if (shouldOverwrite && file) {
            this.handleFileSelected(file, true);
        } else {
            stateManager.setState({
                uploadProgress: 0,
                uploadFilename: null,
            });
        }
    }

    updateUploadProgress(progress) {
        const percent = Math.round(progress * 100);
        stateManager.setState({ uploadProgress: progress });

        if (this.elements.uploadProgressFill) {
            this.elements.uploadProgressFill.style.width = `${percent}%`;
        }
        dom.setText(this.elements.uploadPercentText, percent);
    }

    handleUploadSuccess(result) {
        dom.hide(this.elements.uploadProgress);

        stateManager.setState({
            projectId: result.project_id,
            projectDir: result.project_dir,
            projectName: result.project_id,
            videoInfo: result.video_info,
            vramAnalysis: result.vram_analysis,
            uploadProgress: 0,
            uploadFilename: null,
        });

        stateManager.dispatchEvent(new CustomEvent(EVENTS.UPLOAD_COMPLETE, {
            detail: { projectId: result.project_id },
        }));
    }

    handleUploadError(error) {
        dom.hide(this.elements.uploadProgress);
        stateManager.setState({
            uploadProgress: 0,
            uploadFilename: null,
        });
        stateManager.showError(error.message || 'Upload failed');
    }

    reset() {
        // Reset file input
        if (this.elements.fileInput) {
            this.elements.fileInput.value = '';
        }

        // Hide upload progress
        dom.hide(this.elements.uploadProgress);

        // Reset progress
        if (this.elements.uploadProgressFill) {
            this.elements.uploadProgressFill.style.width = '0%';
        }
        dom.setText(this.elements.uploadPercentText, '0');
    }

    destroy() {
        if (this.elements.browseBtn && this._boundHandlers.onBrowseClick) {
            this.elements.browseBtn.removeEventListener('click', this._boundHandlers.onBrowseClick);
        }
        if (this.elements.fileInput && this._boundHandlers.onFileChange) {
            this.elements.fileInput.removeEventListener('change', this._boundHandlers.onFileChange);
        }
        if (this.elements.dropZone) {
            this.elements.dropZone.removeEventListener('dragover', this._boundHandlers.onDragOver);
            this.elements.dropZone.removeEventListener('dragleave', this._boundHandlers.onDragLeave);
            this.elements.dropZone.removeEventListener('drop', this._boundHandlers.onDrop);
        }
        if (this.elements.overwriteYes && this._boundHandlers.onOverwriteYes) {
            this.elements.overwriteYes.removeEventListener('click', this._boundHandlers.onOverwriteYes);
        }
        if (this.elements.overwriteNo && this._boundHandlers.onOverwriteNo) {
            this.elements.overwriteNo.removeEventListener('click', this._boundHandlers.onOverwriteNo);
        }
        if (this.elements.overwriteCancel && this._boundHandlers.onOverwriteCancel) {
            this.elements.overwriteCancel.removeEventListener('click', this._boundHandlers.onOverwriteCancel);
        }
        this._boundHandlers = {};
        this._pendingFile = null;
        this._pendingProjectName = null;
    }
}
