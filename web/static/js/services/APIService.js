/**
 * APIService - HTTP API client
 *
 * Handles all HTTP requests to the backend API.
 * Provides a clean abstraction over the Fetch API with:
 * - Consistent error handling
 * - Request/response transformation
 * - Retry logic
 * - Type safety
 *
 * Follows Dependency Inversion Principle - high-level code depends
 * on this abstraction, not directly on Fetch API.
 */

import { API } from '../config/constants.js';

export class APIService {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    /**
     * Handle non-OK response by extracting error message
     * @private
     */
    async _handleErrorResponse(response, fallbackMessage = 'Request failed') {
        const error = await response.json().catch(() => ({
            detail: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new Error(error.detail || fallbackMessage);
    }

    /**
     * Make a generic API request
     * @private
     */
    async _request(url, options = {}) {
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(this.baseURL + url, config);

            if (!response.ok) {
                await this._handleErrorResponse(response);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }

            return null;
        } catch (error) {
            console.error(`API request failed: ${url}`, error);
            throw error;
        }
    }

    /**
     * GET request
     */
    async get(url) {
        return this._request(url, { method: 'GET' });
    }

    /**
     * POST request
     */
    async post(url, data = null) {
        return this._request(url, {
            method: 'POST',
            body: data ? JSON.stringify(data) : null,
        });
    }

    /**
     * POST with FormData (for file uploads)
     */
    async postForm(url, formData) {
        const response = await fetch(this.baseURL + url, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            await this._handleErrorResponse(response, 'Upload failed');
        }

        try {
            return await response.json();
        } catch {
            throw new Error('Invalid response format from server');
        }
    }

    // ----- Specific API endpoints -----

    /**
     * Get application configuration
     */
    async getConfig() {
        return this.get(API.CONFIG);
    }

    /**
     * Get system status
     */
    async getSystemStatus() {
        return this.get(API.SYSTEM_STATUS);
    }

    /**
     * Shutdown the system
     */
    async shutdownSystem() {
        return this.post('/api/system/shutdown', {});
    }

    /**
     * Get projects list
     */
    async getProjects() {
        return this.get(API.PROJECTS);
    }

    /**
     * Upload video file
     * @param {File} file - Video file
     * @param {Function} onProgress - Progress callback (0-1)
     */
    async uploadVideo(file, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress(e.loaded / e.total);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        resolve(data);
                    } catch (error) {
                        reject(new Error('Invalid response format'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.detail || `Upload failed: ${xhr.status}`));
                    } catch {
                        reject(new Error(`Upload failed: ${xhr.status}`));
                    }
                }
            });

            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });

            xhr.addEventListener('abort', () => {
                reject(new Error('Upload cancelled'));
            });

            const formData = new FormData();
            formData.append('file', file);

            xhr.open('POST', this.baseURL + API.UPLOAD);
            xhr.send(formData);
        });
    }

    /**
     * Start processing pipeline
     * @param {string} projectId - Project ID
     * @param {Object} config - Pipeline configuration
     */
    async startProcessing(projectId, config) {
        return this.post(API.PROJECT_START(projectId), config);
    }

    /**
     * Cancel processing
     * @param {string} projectId - Project ID
     */
    async cancelProcessing(projectId) {
        return this.post(API.PROJECT_CANCEL(projectId));
    }

    /**
     * Get project outputs
     * @param {string} projectId - Project ID
     */
    async getProjectOutputs(projectId) {
        return this.get(API.PROJECT_OUTPUTS(projectId));
    }

    /**
     * Open project folder
     * @param {string} projectId - Project ID
     */
    async openProjectFolder(projectId) {
        return this.post(API.PROJECT_OPEN_FOLDER(projectId));
    }
}

// Export singleton instance
export const apiService = new APIService();
