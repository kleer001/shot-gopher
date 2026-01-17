/**
 * DOM utility functions
 *
 * Pure functions for DOM manipulation.
 * Provides null-safe operations and common patterns.
 */

import { CSS_CLASSES } from '../config/constants.js';

/**
 * Get element by ID with null safety
 * @param {string} id - Element ID
 * @returns {HTMLElement|null} Element or null
 */
export function getElement(id) {
    return document.getElementById(id);
}

/**
 * Get all elements matching selector
 * @param {string} selector - CSS selector
 * @param {HTMLElement} parent - Parent element (default: document)
 * @returns {NodeList} Elements
 */
export function getElements(selector, parent = document) {
    return parent.querySelectorAll(selector);
}

/**
 * Show element by removing hidden class
 * @param {HTMLElement|string} element - Element or ID
 */
export function show(element) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.classList.remove(CSS_CLASSES.HIDDEN);
    }
}

/**
 * Hide element by adding hidden class
 * @param {HTMLElement|string} element - Element or ID
 */
export function hide(element) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.classList.add(CSS_CLASSES.HIDDEN);
    }
}

/**
 * Toggle element visibility
 * @param {HTMLElement|string} element - Element or ID
 * @returns {boolean} True if now visible, false if hidden
 */
export function toggle(element) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.classList.toggle(CSS_CLASSES.HIDDEN);
        return !el.classList.contains(CSS_CLASSES.HIDDEN);
    }
    return false;
}

/**
 * Set element text content safely
 * @param {HTMLElement|string} element - Element or ID
 * @param {string} text - Text content
 */
export function setText(element, text) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.textContent = text;
    }
}

/**
 * Set element HTML content safely
 * @param {HTMLElement|string} element - Element or ID
 * @param {string} html - HTML content
 */
export function setHTML(element, html) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.innerHTML = html;
    }
}

/**
 * Add CSS class to element
 * @param {HTMLElement|string} element - Element or ID
 * @param {string} className - Class name
 */
export function addClass(element, className) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.classList.add(className);
    }
}

/**
 * Remove CSS class from element
 * @param {HTMLElement|string} element - Element or ID
 * @param {string} className - Class name
 */
export function removeClass(element, className) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.classList.remove(className);
    }
}

/**
 * Toggle CSS class on element
 * @param {HTMLElement|string} element - Element or ID
 * @param {string} className - Class name
 * @returns {boolean} True if class is now present
 */
export function toggleClass(element, className) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        return el.classList.toggle(className);
    }
    return false;
}

/**
 * Create element with attributes and children
 * @param {string} tag - HTML tag name
 * @param {Object} attributes - Element attributes
 * @param {Array<HTMLElement|string>} children - Child elements or text
 * @returns {HTMLElement} Created element
 */
export function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);

    // Set attributes
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'dataset') {
            Object.entries(value).forEach(([dataKey, dataValue]) => {
                element.dataset[dataKey] = dataValue;
            });
        } else if (key.startsWith('on') && typeof value === 'function') {
            // Event listeners
            element.addEventListener(key.substring(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    });

    // Add children
    children.forEach(child => {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else if (child instanceof HTMLElement) {
            element.appendChild(child);
        }
    });

    return element;
}

/**
 * Clear all children from element
 * @param {HTMLElement|string} element - Element or ID
 */
export function clearChildren(element) {
    const el = typeof element === 'string' ? getElement(element) : element;
    if (el) {
        el.innerHTML = '';
    }
}

/**
 * Check if element exists and is visible
 * @param {HTMLElement|string} element - Element or ID
 * @returns {boolean} True if exists and visible
 */
export function isVisible(element) {
    const el = typeof element === 'string' ? getElement(element) : element;
    return el && !el.classList.contains(CSS_CLASSES.HIDDEN);
}
