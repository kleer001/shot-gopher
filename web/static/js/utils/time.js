/**
 * Time utility functions
 *
 * Pure functions for time formatting and calculation.
 * No side effects, easy to test.
 */

/**
 * Format seconds into MM:SS or HH:MM:SS
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string
 */
export function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    return `${minutes}:${String(secs).padStart(2, '0')}`;
}

/**
 * Format seconds into human-readable duration
 * @param {number} seconds - Time in seconds
 * @returns {string} Human-readable duration
 */
export function formatDuration(seconds) {
    if (seconds < 60) {
        return `${Math.round(seconds)}s`;
    }

    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
        return `${minutes}m ${Math.round(seconds % 60)}s`;
    }

    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m`;
}

/**
 * Calculate elapsed time since a start time
 * @param {number} startTime - Start time in milliseconds
 * @returns {number} Elapsed time in seconds
 */
export function getElapsedTime(startTime) {
    if (!startTime) return 0;
    return Math.floor((Date.now() - startTime) / 1000);
}

/**
 * Estimate remaining time based on progress
 * @param {number} progress - Progress (0-1)
 * @param {number} elapsedSeconds - Elapsed time in seconds
 * @returns {number|null} Estimated remaining seconds, or null if can't estimate
 */
export function estimateRemainingTime(progress, elapsedSeconds) {
    if (progress <= 0 || progress >= 1) {
        return null;
    }

    const totalEstimate = elapsedSeconds / progress;
    return Math.round(totalEstimate - elapsedSeconds);
}
