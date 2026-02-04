/**
 * ThemeManager - Color palette management
 *
 * Manages application color themes with:
 * - Runtime CSS variable switching
 * - Persistent theme preference via localStorage
 * - Event-driven theme changes
 * - Support for multiple expandable palettes
 *
 * Follows the Single Responsibility Principle by only managing themes.
 */

import { PALETTES, DEFAULT_PALETTE, THEME_STORAGE_KEY, EVENTS } from '../config/constants.js';

export class ThemeManager extends EventTarget {
    constructor() {
        super();
        this._currentPaletteId = DEFAULT_PALETTE;
        this._isPopupOpen = false;
    }

    /**
     * Initialize the theme manager
     * Loads saved preference and applies it
     */
    init() {
        const savedTheme = this._loadSavedTheme();
        this.applyPalette(savedTheme);
        this._bindPopupEvents();
    }

    /**
     * Get available palettes
     * @returns {Array<Object>} Array of palette objects
     */
    getPalettes() {
        return Object.values(PALETTES);
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
        return PALETTES[this._currentPaletteId] || PALETTES[DEFAULT_PALETTE];
    }

    /**
     * Apply a palette by ID
     * @param {string} paletteId - Palette identifier
     */
    applyPalette(paletteId) {
        const palette = PALETTES[paletteId];
        if (!palette) {
            console.warn(`Unknown palette: ${paletteId}, falling back to ${DEFAULT_PALETTE}`);
            this.applyPalette(DEFAULT_PALETTE);
            return;
        }

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
            if (saved && PALETTES[saved]) {
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
