/**
 * UploadController - Manages file upload UI and logic
 *
 * Responsibilities:
 * - Handle drag-and-drop file uploads
 * - Manage upload progress display
 * - Validate file types
 * - Update video info after upload
 *
 * Follows Single Responsibility Principle by only handling upload UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { ELEMENTS, UPLOAD } from '../config/constants.js';

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
            videoInfo: dom.getElement('video-info'),
            videoName: dom.getElement('video-name'),
            videoResolution: dom.getElement('video-resolution'),
            videoFrames: dom.getElement('video-frames'),
            videoFps: dom.getElement('video-fps'),
        };

        this._boundHandlers = {};
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

    async handleFileSelected(file) {
        // Validate file type - handle edge cases (no extension, trailing dot)
        const lastDotIndex = file.name.lastIndexOf('.');
        const ext = lastDotIndex > 0 ? file.name.slice(lastDotIndex).toLowerCase() : '';
        const config = stateManager.get('config');
        const supportedFormats = config?.supportedVideoFormats || UPLOAD.SUPPORTED_FORMATS;

        if (!ext || !supportedFormats.includes(ext)) {
            const displayExt = ext || '(no extension)';
            stateManager.showError(`Unsupported file type: ${displayExt}`);
            return;
        }

        // Show upload progress
        dom.show(this.elements.uploadProgress);
        dom.setText(this.elements.uploadFilename, file.name);
        stateManager.setState({ uploadFilename: file.name, uploadProgress: 0 });

        try {
            // Upload file
            const result = await apiService.uploadVideo(file, (progress) => {
                this.updateUploadProgress(progress);
            });

            // Upload successful
            this.handleUploadSuccess(result);
        } catch (error) {
            this.handleUploadError(error);
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
        // Hide upload progress
        dom.hide(this.elements.uploadProgress);

        // Update state with video info and VRAM analysis
        stateManager.setState({
            projectId: result.project_id,
            projectDir: result.project_dir,
            projectName: result.project_id,
            videoInfo: result.video_info,
            vramAnalysis: result.vram_analysis,
            uploadProgress: 0,
            uploadFilename: null,
        });

        // Validate and show video info
        if (result.video_info && result.video_info.resolution) {
            this.displayVideoInfo(result.video_info, result.project_id);
        } else {
            // Show warning but continue
            stateManager.showError('Could not extract video info, but upload succeeded');
        }

        // Show config form
        const configForm = dom.getElement(ELEMENTS.CONFIG_FORM);
        if (configForm) {
            dom.show(configForm);
        }
    }

    displayVideoInfo(info, filename) {
        if (!this.elements.videoInfo) return;

        // Validate info has required properties
        if (!info || !info.resolution || !info.fps || info.frame_count === undefined) {
            // Show default values for missing info
            dom.setText(this.elements.videoName, filename);
            dom.setText(this.elements.videoResolution, 'Unknown');
            dom.setText(this.elements.videoFrames, 'Unknown');
            dom.setText(this.elements.videoFps, 'Unknown');
            dom.show(this.elements.videoInfo);
            return;
        }

        dom.setText(this.elements.videoName, filename);
        dom.setText(this.elements.videoResolution, `${info.resolution[0]}x${info.resolution[1]}`);
        dom.setText(this.elements.videoFrames, info.frame_count);
        dom.setText(this.elements.videoFps, info.fps.toFixed(2));
        dom.show(this.elements.videoInfo);
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

        // Hide upload progress and video info
        dom.hide(this.elements.uploadProgress);
        dom.hide(this.elements.videoInfo);

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
        this._boundHandlers = {};
    }
}
