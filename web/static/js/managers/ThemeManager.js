/**
 * ThemeManager - Color palette management
 *
 * Manages application color themes with:
 * - Runtime CSS variable switching
 * - Persistent theme preference via localStorage
 * - Event-driven theme changes
 * - Dynamic palette loading from API
 *
 * Follows the Single Responsibility Principle by only managing themes.
 */

import { DEFAULT_PALETTE, THEME_STORAGE_KEY, EVENTS, API_ENDPOINTS } from '../config/constants.js';

const FALLBACK_PALETTE = {
    id: 'dark',
    name: 'Dark',
    icon: '\u{1F319}',
    colors: {
        '--bg': '#0a0e27',
        '--bg-card': '#131b3a',
        '--bg-hover': '#1a2547',
        '--border': '#1e2d5f',
        '--text': '#e0e7ff',
        '--text-dim': '#6b7db8',
        '--accent': '#3b82f6',
        '--accent-glow': 'rgba(59, 130, 246, 0.3)',
        '--success': '#10b981',
        '--warning': '#f59e0b',
        '--danger': '#ef4444',
    },
};

export class ThemeManager extends EventTarget {
    constructor() {
        super();
        this._currentPaletteId = DEFAULT_PALETTE;
        this._isPopupOpen = false;
        this._palettes = {};
        this._isLoaded = false;
    }

    /**
     * Initialize the theme manager
     * Loads saved preference and applies it
     */
    async init() {
        const savedTheme = this._loadSavedTheme();
        this._currentPaletteId = savedTheme;

        await this._loadPalettes();

        this.applyPalette(savedTheme);
        this._bindPopupEvents();
    }

    /**
     * Load palettes from API
     * @private
     */
    async _loadPalettes() {
        try {
            const response = await fetch(API_ENDPOINTS.PALETTES);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const paletteList = data.palettes || [];

            this._palettes = {};
            paletteList.forEach(palette => {
                this._palettes[palette.id] = palette;
            });

            this._isLoaded = true;
        } catch (error) {
            console.warn('Failed to load palettes from API, using fallback:', error);
            this._palettes = { [FALLBACK_PALETTE.id]: FALLBACK_PALETTE };
            this._isLoaded = true;
        }
    }

    /**
     * Get available palettes
     * @returns {Array<Object>} Array of palette objects
     */
    getPalettes() {
        return Object.values(this._palettes);
    }

    /**
     * Get current palette ID
     * @returns {string} Current palette ID
     */
    getCurrentPaletteId() {
        return this._currentPaletteId;
    }

    /**
     * Get current palette object
     * @returns {Object} Current palette
     */
    getCurrentPalette() {
        return this._palettes[this._currentPaletteId] || FALLBACK_PALETTE;
    }

    /**
     * Apply a palette by ID
     * @param {string} paletteId - Palette identifier
     */
    applyPalette(paletteId) {
        const palette = this._palettes[paletteId] || FALLBACK_PALETTE;

        const root = document.documentElement;
        Object.entries(palette.colors).forEach(([property, value]) => {
            root.style.setProperty(property, value);
        });

        this._currentPaletteId = paletteId;
        this._savePalettePreference(paletteId);
        this._updateActiveState();

        this.dispatchEvent(new CustomEvent(EVENTS.THEME_CHANGED, {
            detail: { paletteId, palette },
        }));
    }

    /**
     * Toggle theme picker popup visibility
     */
    togglePopup() {
        const popup = document.getElementById('theme-picker-popup');
        if (!popup) return;

        this._isPopupOpen = !this._isPopupOpen;
        popup.classList.toggle('hidden', !this._isPopupOpen);

        if (this._isPopupOpen) {
            this._updateActiveState();
        }
    }

    /**
     * Close the theme picker popup
     */
    closePopup() {
        const popup = document.getElementById('theme-picker-popup');
        if (popup) {
            popup.classList.add('hidden');
        }
        this._isPopupOpen = false;
    }

    /**
     * Render palette options into the popup
     */
    renderPaletteOptions() {
        const container = document.getElementById('theme-options');
        if (!container) return;

        container.innerHTML = '';
        const palettes = this.getPalettes();

        palettes.forEach(palette => {
            const option = document.createElement('button');
            option.type = 'button';
            option.className = 'theme-option';
            option.dataset.paletteId = palette.id;
            option.innerHTML = `
                <span class="theme-option-icon">${palette.icon}</span>
                <span class="theme-option-name">${palette.name}</span>
                <span class="theme-option-preview">
                    <span class="preview-swatch" style="background: ${palette.colors['--bg']}"></span>
                    <span class="preview-swatch" style="background: ${palette.colors['--bg-card']}"></span>
                    <span class="preview-swatch" style="background: ${palette.colors['--accent']}"></span>
                </span>
            `;

            option.addEventListener('click', () => {
                this.applyPalette(palette.id);
            });

            container.appendChild(option);
        });

        this._updateActiveState();
    }

    /**
     * Load saved theme preference from localStorage
     * @returns {string} Saved palette ID or default
     * @private
     */
    _loadSavedTheme() {
        try {
            const saved = localStorage.getItem(THEME_STORAGE_KEY);
            if (saved) {
                return saved;
            }
        } catch (error) {
            console.warn('Failed to load theme preference:', error);
        }
        return DEFAULT_PALETTE;
    }

    /**
     * Save palette preference to localStorage
     * @param {string} paletteId - Palette ID to save
     * @private
     */
    _savePalettePreference(paletteId) {
        try {
            localStorage.setItem(THEME_STORAGE_KEY, paletteId);
        } catch (error) {
            console.warn('Failed to save theme preference:', error);
        }
    }

    /**
     * Update active state on palette options
     * @private
     */
    _updateActiveState() {
        const options = document.querySelectorAll('.theme-option');
        options.forEach(option => {
            const isActive = option.dataset.paletteId === this._currentPaletteId;
            option.classList.toggle('active', isActive);
        });

        const currentIcon = document.getElementById('current-theme-icon');
        if (currentIcon) {
            const palette = this.getCurrentPalette();
            currentIcon.textContent = palette.icon;
        }
    }

    /**
     * Bind popup event listeners
     * @private
     */
    _bindPopupEvents() {
        const trigger = document.getElementById('theme-picker-trigger');
        if (trigger) {
            trigger.addEventListener('click', () => this.togglePopup());
        }

        document.addEventListener('click', (event) => {
            if (!this._isPopupOpen) return;

            const popup = document.getElementById('theme-picker-popup');
            const trigger = document.getElementById('theme-picker-trigger');

            if (popup && trigger) {
                const clickedInside = popup.contains(event.target) || trigger.contains(event.target);
                if (!clickedInside) {
                    this.closePopup();
                }
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape' && this._isPopupOpen) {
                this.closePopup();
            }
        });
    }
}

export const themeManager = new ThemeManager();
