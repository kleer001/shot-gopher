/**
 * ProjectsController - Manages projects list UI
 *
 * Responsibilities:
 * - Load and display projects list
 * - Handle project selection
 * - Auto-refresh projects
 *
 * Follows Single Responsibility Principle by only handling projects UI.
 */

import { stateManager } from '../managers/StateManager.js';
import { apiService } from '../services/APIService.js';
import * as dom from '../utils/dom.js';
import { ELEMENTS } from '../config/constants.js';

export class ProjectsController {
    constructor() {
        this.elements = {
            projectsList: dom.getElement(ELEMENTS.PROJECTS_LIST),
        };

        this.refreshInterval = null;
        this.loadProjects();
        this.startAutoRefresh();
    }

    async loadProjects() {
        if (!this.elements.projectsList) return;

        try {
            const data = await apiService.getProjects();

            if (data.projects && data.projects.length > 0) {
                this.displayProjects(data.projects.slice(0, 5));
            } else {
                dom.setHTML(this.elements.projectsList, '<p class="no-projects">(none yet)</p>');
            }
        } catch (error) {
            console.error('Failed to load projects:', error);
            dom.setHTML(this.elements.projectsList, '<p class="no-projects">Failed to load</p>');
        }
    }

    displayProjects(projects) {
        const html = projects.map(proj => {
            const safeName = dom.escapeHTML(proj.name || proj.project_id);
            const safeId = dom.escapeHTML(proj.project_id);
            const safeDir = dom.escapeHTML(proj.project_dir || '');
            const safeStatus = dom.escapeHTML(proj.status || 'unknown');
            return `
            <div class="project-item" data-id="${safeId}" data-dir="${safeDir}">
                <span class="project-name">${safeName}</span>
                <span class="project-status ${safeStatus}">${safeStatus}</span>
            </div>
        `;
        }).join('');

        dom.setHTML(this.elements.projectsList, html);

        // Add click handlers
        const projectItems = this.elements.projectsList.querySelectorAll('.project-item');
        projectItems.forEach(item => {
            item.addEventListener('click', () => {
                this.handleProjectClick(item);
            });
        });

        // Update state
        stateManager.setState({ projects });
    }

    handleProjectClick(item) {
        const projectId = item.dataset.id;
        const projectDir = item.dataset.dir;

        console.log('Project clicked:', projectId);

        // For now, just log - could implement project loading here
        // In the future, this could load the project for viewing outputs
    }

    startAutoRefresh() {
        // Refresh projects list every 10 seconds
        this.refreshInterval = setInterval(() => {
            this.loadProjects();
        }, 10000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    destroy() {
        this.stopAutoRefresh();
    }
}
