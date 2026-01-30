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
    ERROR_TOAST_DURATION: 5000,
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

    // Config
    CONFIG_FORM: 'config-form',
    ROTO_PROMPT: 'roto-prompt',
    SKIP_EXISTING: 'skip-existing',

    // Processing
    PROCESSING_PANEL: 'processing-panel',
    PROGRESS_PERCENT: 'progress-percent',
    CURRENT_STAGE_NAME: 'current-stage-name',
    LOG_OUTPUT: 'log-output',

    // Projects
    PROJECTS_LIST: 'projects-list',

    // Project Detail
    PROJECT_DETAIL_PANEL: 'project-detail-panel',
    DETAIL_PROJECT_NAME: 'detail-project-name',
    DETAIL_STAGES: 'detail-stages',
    STAGES_COUNTER: 'stages-counter',
    VRAM_INFO_SECTION: 'vram-info-section',
    VRAM_INFO: 'vram-info',
    DETAIL_VIDEO_SECTION: 'detail-video-section',
    DETAIL_VIDEO_INFO: 'detail-video-info',
    OPEN_FOLDER_BTN: 'open-folder-btn',
    DELETE_PROJECT_BTN: 'delete-project-btn',
    REPROCESS_BTN: 'reprocess-btn',

    // System
    SYSTEM_STATUS: 'system-status',
    ERROR_TOAST: 'error-toast',
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
    ERROR: 'error',
    CONFIG_LOADED: 'configLoaded',
};
