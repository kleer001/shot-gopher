/**
 * Main Application Entry Point
 *
 * This is the main entry point for the VFX Pipeline web application.
 * It initializes all controllers and services, wiring everything together.
 *
 * Architecture:
 * - Services: Handle data and communication (API, WebSocket)
 * - Managers: Handle state management
 * - Controllers: Handle UI logic and user interactions
 * - Utils: Provide reusable functions
 *
 * The application follows SOLID principles with clear separation of concerns.
 */

import { stateManager } from './managers/StateManager.js';
import { themeManager } from './managers/ThemeManager.js';
import { apiService } from './services/APIService.js';
import { wsService } from './services/WebSocketService.js';
import { UploadController } from './controllers/UploadController.js';
import { ProjectsController } from './controllers/ProjectsController.js';
import { SystemController } from './controllers/SystemController.js';
import { BugReportController } from './controllers/BugReportController.js';

/**
 * Application class - manages application lifecycle
 */
class Application {
    constructor() {
        this.controllers = {};
        this.isInitialized = false;
    }

    /**
     * Initialize the application
     */
    async init() {
        if (this.isInitialized) {
            console.warn('Application already initialized');
            return;
        }

        console.log('Initializing VFX Pipeline application...');

        try {
            // Load configuration first
            await this.loadConfiguration();

            // Initialize controllers
            await this.initializeControllers();

            // Mark as initialized
            this.isInitialized = true;

            console.log('Application initialized successfully');
        } catch (error) {
            console.error('Failed to initialize application:', error);
            stateManager.showError('Failed to initialize application');
        }
    }

    /**
     * Load configuration from API
     */
    async loadConfiguration() {
        console.log('Loading configuration...');

        try {
            const config = await apiService.getConfig();
            stateManager.setConfig(config);
            console.log('Configuration loaded:', config);
        } catch (error) {
            console.error('Failed to load configuration:', error);
            // Continue with default configuration
            stateManager.setConfig({
                stages: {},
                presets: {},
                supportedVideoFormats: ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'],
            });
        }
    }

    /**
     * Initialize all controllers
     */
    async initializeControllers() {
        console.log('Initializing controllers...');

        // Initialize theme manager first (loads palettes from API and applies saved theme)
        await themeManager.init();
        themeManager.renderPaletteOptions();
        console.log('Theme manager initialized');

        // System controller (checks ComfyUI status, handles errors)
        this.controllers.system = new SystemController();

        // Upload controller (handles file uploads)
        this.controllers.upload = new UploadController();

        // Projects controller (handles projects list and detail view)
        this.controllers.projects = new ProjectsController();

        // Bug report controller (handles bug report button and dialog)
        this.controllers.bugReport = new BugReportController();

        console.log('Controllers initialized');
    }

    /**
     * Reset application to initial state
     */
    reset() {
        // Reset state
        stateManager.reset();

        // Reset controllers
        Object.values(this.controllers).forEach(controller => {
            if (controller.reset) {
                controller.reset();
            }
        });
    }

    /**
     * Destroy application and clean up
     */
    destroy() {
        // Disconnect WebSocket
        wsService.disconnect();

        // Destroy controllers
        Object.values(this.controllers).forEach(controller => {
            if (controller.destroy) {
                controller.destroy();
            }
        });

        this.controllers = {};
        this.isInitialized = false;
    }
}

// Create global application instance
const app = new Application();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        app.init();
    });
} else {
    // DOM already loaded
    app.init();
}

// Export for debugging
window.app = app;
window.stateManager = stateManager;
window.themeManager = themeManager;
window.apiService = apiService;
window.wsService = wsService;

// Log application info
console.log('VFX Pipeline Web Application');
console.log('Version: 1.0.0');
console.log('Architecture: Modular ES6 with SOLID principles');
