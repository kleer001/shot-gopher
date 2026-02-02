/**
 * Application constants and configuration
 *
 * This module provides constant values used throughout the application.
 * Configuration that changes should be loaded from the API.
 */

/**
 * API endpoints
 */
export const API = {
    CONFIG: '/api/config',
    UPLOAD: '/api/upload',
    PROJECTS: '/api/projects',
    SYSTEM_STATUS: '/api/system/status',
    PROJECT_OUTPUTS: (id) => `/api/projects/${id}/outputs`,
    PROJECT_CANCEL: (id) => `/api/projects/${id}/stop`,  // Backend uses /stop endpoint
    PROJECT_START: (id) => `/api/projects/${id}/start`,
    PROJECT_JOB: (id) => `/api/projects/${id}/job`,
    PROJECT_OPEN_FOLDER: (id) => `/api/projects/${id}/open-folder`,
    PROJECT_VRAM: (id) => `/api/projects/${id}/vram`,
    PROJECT_VIDEO_INFO: (id) => `/api/projects/${id}/video-info`,
    PROJECT_INTERACTIVE_COMPLETE: (id) => `/api/projects/${id}/interactive-complete`,
};

/**
 * WebSocket configuration
 */
export const WEBSOCKET = {
    PING_INTERVAL: 25000,
    RECONNECT_DELAY: 2000,
    MAX_RECONNECT_ATTEMPTS: 5,
};

/**
 * UI configuration
 */
export const UI = {
    MAX_PROJECTS_DISPLAYED: 5,
    PROGRESS_UPDATE_INTERVAL: 100,
    PROCESSING_POLL_INTERVAL: 1000,
    ERROR_TOAST_DURATION: 5000,
    BUTTON_RESET_DELAY: 2000,
    BUTTON_FAILURE_DELAY: 3000,
    SIGNAL_SENT_DELAY: 1000,
    PROJECTS_REFRESH_INTERVAL: 10000,
    SHUTDOWN_MESSAGE_DELAY: 2000,
    SYSTEM_STATUS_INTERVAL: 30000,
};

/**
 * File upload configuration
 */
export const UPLOAD = {
    CHUNK_SIZE: 1024 * 1024, // 1MB chunks
    SUPPORTED_FORMATS: ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'],
};

/**
 * DOM element IDs (for reference)
 */
export const ELEMENTS = {
    // Upload
    DROP_ZONE: 'drop-zone',
    FILE_INPUT: 'file-input',
    BROWSE_BTN: 'browse-btn',
    UPLOAD_PROGRESS: 'upload-progress',

    // Projects
    PROJECTS_LIST: 'projects-list',

    // Project Detail
    PROJECT_DETAIL_PANEL: 'project-detail-panel',
    DETAIL_PROJECT_NAME: 'detail-project-name',
    DETAIL_STAGES: 'detail-stages',
    STAGES_COUNTER: 'stages-counter',
    VRAM_INFO_SECTION: 'vram-info-section',
    VRAM_INFO: 'vram-info',
    OPEN_FOLDER_BTN: 'open-folder-btn',
    DELETE_PROJECT_BTN: 'delete-project-btn',
    REPROCESS_BTN: 'reprocess-btn',
    INTERACTIVE_COMPLETE_BTN: 'interactive-complete-btn',

    // System
    SYSTEM_STATUS: 'system-status',
    ERROR_TOAST: 'error-toast',

    // Overwrite Dialog
    OVERWRITE_DIALOG: 'overwrite-dialog',
    OVERWRITE_PROJECT_NAME: 'overwrite-project-name',
    OVERWRITE_YES: 'overwrite-yes',
    OVERWRITE_NO: 'overwrite-no',
    OVERWRITE_CANCEL: 'overwrite-cancel',
};

/**
 * CSS class names
 */
export const CSS_CLASSES = {
    HIDDEN: 'hidden',
    ACTIVE: 'active',
    ONLINE: 'online',
    OFFLINE: 'offline',
    PROCESSING: 'processing',
    COMPLETED: 'completed',
    FAILED: 'failed',
};

/**
 * Event names
 */
export const EVENTS = {
    STATE_CHANGED: 'stateChanged',
    PROGRESS_UPDATE: 'progressUpdate',
    UPLOAD_PROGRESS: 'uploadProgress',
    UPLOAD_COMPLETE: 'uploadComplete',
    ERROR: 'error',
    CONFIG_LOADED: 'configLoaded',
    PIPELINE_COMPLETE: 'pipelineComplete',
    PIPELINE_FAILED: 'pipelineFailed',
};
