# 🌐 Roadmap 3: Web UI Frontend

**Goal:** Build browser-based UI (presentation layer only - no business logic)

**Status:** 🟡 70% Complete (Core UI: 90%, Testing/Polish: 20%)

**Dependencies:** Originally planned after Roadmap 2 (API), but developed in parallel

---

## Implementation Status

### ✅ Completed (70%)
- **Phase 3A**: HTML templates with components (`web/templates/`)
- **Phase 3B**: Multiple responsive CSS layouts (`web/static/css/`)
- **Phase 3C**: API client abstraction (`APIService.js`, `WebSocketService.js`)
- **Phase 3D**: Modular ES6 controllers with SOLID principles
  - Upload, Config, Processing, Projects, System controllers
  - StateManager for application state
  - DOM and time utilities
- **Architecture**: "Dumb UI" pattern - zero business logic in frontend

### ⚪ Remaining (30%)
- **Phase 3E**: Comprehensive UI testing suite
- **Phase 3E**: Accessibility improvements (ARIA, keyboard nav)
- **Phase 3E**: Cross-browser compatibility testing
- **Phase 3F**: One-click startup script
- **Polish**: Performance optimization (lazy loading, code splitting)

---

## Overview

This roadmap builds the web-based user interface that consumes the API. The core UI is **fully functional** with modular ES6 architecture and follows SOLID principles. The frontend is a **pure presentation layer** - it makes API calls and renders responses. Zero business logic.

### Why UI-Last?

1. **Backend is validated** - API proven to work before building UI
2. **Clear contracts** - OpenAPI spec defines exactly what UI can do
3. **Easier testing** - UI tests can mock API responses
4. **Faster iteration** - Can rebuild UI without touching backend
5. **Multiple UIs possible** - Could build mobile app, CLI, etc. using same API

### Architecture Principles

**Presentation Layer Only:**
- **Makes API calls** - All requests through API client abstraction
- **Renders data** - Displays what API returns (no calculations)
- **Handles user input** - Validates locally, sends to API
- **NO business logic** - No progress calculations, no file I/O, no orchestration

### Technology Stack

```
Frontend Components:
├── HTML Templates (Jinja2)
│   └── Server-side rendering for initial page load
├── CSS (Custom + responsive)
│   └── Clean, artist-friendly styling
└── JavaScript (Vanilla ES6+)
    ├── API Client (abstraction over fetch)
    ├── WebSocket Client (real-time updates)
    └── UI controllers (event handlers, rendering)
```

### Data Flow

```
User Action
    ↓
JavaScript Event Handler
    ↓
API Client (abstraction)
    ↓
HTTP/WebSocket Request
    ↓
[Roadmap 2 API Backend]
    ↓
Response (JSON)
    ↓
JavaScript Rendering
    ↓
DOM Update (user sees result)
```

**Key Principle:** Frontend just renders API data. All logic stays in backend.

---

## Phase 3A: HTML Templates & Structure ✅

**Status:** 100% Complete - Fully implemented with reusable components

**Goal:** Server-side rendered HTML pages

### Deliverables
- Base template with layout
- Dashboard page template
- Project creation form template
- Processing/progress view template
- Results view template

### Tasks

#### Task 3A.1: Create Base Template
**File:** `web/ui/templates/base.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}VFX Ingest Platform{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/styles.css">
    {% block extra_css %}{% endblock %}
</head>
<body>
    <header class="header">
        <div class="container">
            <h1 class="logo">VFX Ingest Platform</h1>
            <nav class="nav">
                <a href="/dashboard" class="nav-link">Dashboard</a>
                <a href="/projects/new" class="nav-link">New Project</a>
                <button id="shutdown-btn" class="nav-link nav-button">Shutdown</button>
            </nav>
        </div>
    </header>

    <main class="main">
        <div class="container">
            {% block content %}{% endblock %}
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <p>VFX Ingest Platform v1.0 | Docker Mode</p>
        </div>
    </footer>

    <!-- Core JavaScript -->
    <script src="/static/js/api-client.js"></script>
    <script src="/static/js/common.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

**Success Criteria:**
- [ ] Base template renders without errors
- [ ] Navigation works across all pages
- [ ] Responsive layout (mobile, tablet, desktop)
- [ ] Clean HTML5 semantic markup

---

#### Task 3A.2: Create Dashboard Template
**File:** `web/ui/templates/dashboard.html`

```html
{% extends "base.html" %}

{% block title %}Dashboard - VFX Ingest Platform{% endblock %}

{% block content %}
<div class="dashboard">
    <h2 class="page-title">Projects</h2>

    <!-- Projects loaded via JavaScript from API -->
    <div id="project-grid" class="project-grid">
        <div class="loading">Loading projects...</div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="/static/js/dashboard.js"></script>
{% endblock %}
```

**Success Criteria:**
- [ ] Template extends base correctly
- [ ] Loading state shown before API data loads
- [ ] Grid container ready for dynamic content

---

#### Task 3A.3: Create Processing View Template
**File:** `web/ui/templates/processing.html`

```html
{% extends "base.html" %}

{% block title %}{{ project_name }} - Processing{% endblock %}

{% block content %}
<!-- Hidden input for JavaScript to access project name -->
<input type="hidden" id="project-name" value="{{ project_name }}">

<div class="processing">
    <h2 class="page-title">{{ project_name }} - Processing</h2>

    <div class="progress-section">
        <div class="progress-item">
            <h3>Current Stage: <span id="current-stage">-</span></h3>
            <div class="progress-bar">
                <div class="progress-fill" id="stage-progress" style="width: 0%"></div>
            </div>
            <p id="stage-status" class="status-text">Starting...</p>
        </div>
    </div>

    <div class="stage-list">
        <h3>Stages</h3>
        <ul id="stages-list">
            <!-- Populated by JavaScript -->
        </ul>
    </div>

    <div class="actions">
        <button id="stop-btn" class="btn btn-danger">Stop Processing</button>
    </div>

    <div class="logs-section">
        <h3>
            Live Logs
            <button id="toggle-logs" class="btn-text">Show</button>
        </h3>
        <pre id="log-output" class="logs" style="display: none;"></pre>
    </div>
</div>
{% endblock %}

{% block extra_js %}
<script src="/static/js/processing.js"></script>
{% endblock %}
```

**Success Criteria:**
- [ ] Template receives project_name from backend route
- [ ] All interactive elements have IDs for JavaScript
- [ ] Accessible markup (ARIA labels where needed)

---

### Phase 3A Exit Criteria

- [ ] All HTML templates created
- [ ] Templates extend base correctly
- [ ] Semantic HTML5 markup
- [ ] Accessibility considerations (ARIA, semantic tags)
- [ ] Ready for styling and JavaScript

---

## Phase 3B: CSS Styling ✅

**Status:** 100% Complete - Four complete responsive layouts implemented

**Goal:** Clean, professional, artist-friendly styling

### Deliverables
- Base styles and CSS variables
- Component styles (cards, buttons, forms)
- Layout styles (grid, flexbox)
- Responsive design (mobile, tablet, desktop)

### Tasks

#### Task 3B.1: CSS Variables & Base Styles
**File:** `web/ui/static/css/styles.css`

```css
/* CSS Variables (Design Tokens) */
:root {
    /* Colors */
    --color-primary: #3498db;
    --color-primary-dark: #2980b9;
    --color-success: #2ecc71;
    --color-danger: #e74c3c;
    --color-warning: #f39c12;
    --color-background: #f5f5f5;
    --color-surface: #ffffff;
    --color-text: #333333;
    --color-text-muted: #888888;
    --color-border: #dddddd;

    /* Spacing */
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;

    /* Typography */
    --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-size-sm: 14px;
    --font-size-base: 16px;
    --font-size-lg: 18px;
    --font-size-xl: 24px;

    /* Borders */
    --border-radius: 8px;
    --border-width: 1px;

    /* Shadows */
    --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 8px rgba(0, 0, 0, 0.15);
    --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.2);
}

/* Reset */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

/* Base Styles */
body {
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    line-height: 1.6;
    color: var(--color-text);
    background-color: var(--color-background);
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

/* Typography */
h1, h2, h3 {
    margin-bottom: var(--spacing-md);
    font-weight: 600;
}

h1 { font-size: var(--font-size-xl); }
h2 { font-size: var(--font-size-lg); }
h3 { font-size: var(--font-size-base); }
```

**Success Criteria:**
- [ ] CSS variables defined for consistency
- [ ] Base reset applied
- [ ] Typography hierarchy established
- [ ] Easy to maintain and extend

---

#### Task 3B.2: Component Styles
**File:** `web/ui/static/css/styles.css` (continued)

```css
/* Buttons */
.btn {
    display: inline-block;
    padding: var(--spacing-sm) var(--spacing-lg);
    font-size: var(--font-size-base);
    font-weight: 500;
    text-align: center;
    text-decoration: none;
    background-color: var(--color-primary);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    cursor: pointer;
    transition: background-color 0.2s ease;
}

.btn:hover {
    background-color: var(--color-primary-dark);
}

.btn-danger {
    background-color: var(--color-danger);
}

.btn-danger:hover {
    background-color: #c0392b;
}

/* Project Cards */
.project-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: var(--spacing-lg);
    margin-top: var(--spacing-lg);
}

.project-card {
    background: var(--color-surface);
    border-radius: var(--border-radius);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease;
}

.project-card:hover {
    box-shadow: var(--shadow-md);
}

/* Progress Bars */
.progress-bar {
    width: 100%;
    height: 30px;
    background-color: var(--color-border);
    border-radius: calc(var(--border-radius) * 2);
    overflow: hidden;
    margin: var(--spacing-sm) 0;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--color-primary), var(--color-success));
    transition: width 0.3s ease;
}

/* Status Indicators */
.status-created { color: var(--color-primary); }
.status-processing { color: var(--color-warning); }
.status-complete { color: var(--color-success); }
.status-failed { color: var(--color-danger); }

/* Logs */
.logs {
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: var(--spacing-md);
    border-radius: var(--border-radius);
    font-family: 'Courier New', monospace;
    font-size: var(--font-size-sm);
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
}
```

**Success Criteria:**
- [ ] All components styled consistently
- [ ] Hover states for interactive elements
- [ ] Smooth transitions
- [ ] Accessible color contrast (WCAG AA)

---

### Phase 3B Exit Criteria

- [ ] Complete CSS stylesheet
- [ ] Responsive design (mobile, tablet, desktop)
- [ ] Dark mode support (optional)
- [ ] Clean, professional appearance
- [ ] Performance optimized (< 50KB CSS)

---

## Phase 3C: JavaScript API Client ✅

**Status:** 100% Complete - APIService and WebSocketService fully implemented

**Goal:** Abstraction layer over API calls (NO business logic)

### Deliverables
- API client class (wraps fetch)
- Error handling
- Type-safe methods
- WebSocket wrapper

### Tasks

#### Task 3C.1: Create API Client
**File:** `web/ui/static/js/api-client.js`

```javascript
/**
 * API Client - Abstraction over fetch API
 *
 * IMPORTANT: This is JUST an abstraction - no business logic!
 * All logic stays in backend API.
 */

class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    /**
     * Generic request method
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({
                detail: response.statusText
            }));
            throw new Error(error.detail || 'Request failed');
        }

        // Handle 204 No Content
        if (response.status === 204) {
            return null;
        }

        return response.json();
    }

    // ==================
    // Projects API
    // ==================

    async listProjects() {
        return this.request('/api/projects');
    }

    async getProject(name) {
        return this.request(`/api/projects/${name}`);
    }

    async createProject(data) {
        return this.request('/api/projects', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    async uploadVideo(projectName, videoFile) {
        const formData = new FormData();
        formData.append('video', videoFile);

        const response = await fetch(
            `${this.baseUrl}/api/projects/${projectName}/upload-video`,
            {
                method: 'POST',
                body: formData,
            }
        );

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        return response.json();
    }

    async deleteProject(name) {
        return this.request(`/api/projects/${name}`, {
            method: 'DELETE',
        });
    }

    // ==================
    // Pipeline API
    // ==================

    async startJob(projectName, stages) {
        return this.request(`/api/pipeline/projects/${projectName}/start`, {
            method: 'POST',
            body: JSON.stringify({ stages }),
        });
    }

    async stopJob(projectName) {
        return this.request(`/api/pipeline/projects/${projectName}/stop`, {
            method: 'POST',
        });
    }

    async getJobStatus(projectName) {
        return this.request(`/api/pipeline/projects/${projectName}/status`);
    }

    // ==================
    // System API
    // ==================

    async shutdown() {
        return this.request('/api/system/shutdown', {
            method: 'POST',
        });
    }

    // ==================
    // WebSocket
    // ==================

    connectWebSocket(projectName, onMessage, onError = null) {
        const wsUrl = `ws://${window.location.host}/api/pipeline/ws/${projectName}`;
        const ws = new WebSocket(wsUrl);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            onMessage(data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (onError) {
                onError(error);
            }
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
        };

        return ws;
    }
}

// Export singleton instance
const api = new ApiClient();
```

**Success Criteria:**
- [ ] All API endpoints wrapped
- [ ] Consistent error handling
- [ ] WebSocket abstraction included
- [ ] No business logic (just API calls)
- [ ] TypeScript-style JSDoc comments (optional)

---

### Phase 3C Exit Criteria

- [ ] API client complete
- [ ] All endpoints accessible
- [ ] Error handling robust
- [ ] Easy to use in UI code
- [ ] No business logic in client

---

## Phase 3D: UI Logic (Presentation Only) ✅

**Status:** 100% Complete - Modular ES6 controllers with SOLID principles

**Goal:** UI controllers that render API data (NO calculations)

### Deliverables
- Dashboard JavaScript
- Processing view JavaScript
- Form handling JavaScript
- Event handlers

### Tasks

#### Task 3D.1: Dashboard Logic
**File:** `web/ui/static/js/dashboard.js`

```javascript
/**
 * Dashboard UI Logic
 *
 * IMPORTANT: This is PRESENTATION ONLY
 * - Calls API
 * - Renders data
 * - NO calculations, NO business logic
 */

document.addEventListener('DOMContentLoaded', async () => {
    await loadProjects();
});

async function loadProjects() {
    const container = document.getElementById('project-grid');

    try {
        const response = await api.listProjects();
        renderProjects(container, response.projects);
    } catch (error) {
        showError(container, 'Failed to load projects: ' + error.message);
    }
}

function renderProjects(container, projects) {
    // Clear loading state
    container.innerHTML = '';

    // Render each project
    projects.forEach(project => {
        const card = createProjectCard(project);
        container.appendChild(card);
    });

    // Add "New Project" card
    container.appendChild(createNewProjectCard());
}

function createProjectCard(project) {
    const card = document.createElement('div');
    card.className = 'project-card';

    // Just render what API gave us - no transformations
    card.innerHTML = `
        <div class="project-thumbnail">
            <div class="thumbnail-placeholder">
                <span class="project-icon">📁</span>
            </div>
        </div>
        <h3 class="project-name">${escapeHtml(project.name)}</h3>
        <p class="project-status status-${project.status}">${project.status}</p>
        <div class="project-meta">
            <span>${project.stages.length} stages</span>
        </div>
        <div class="project-actions">
            <a href="/projects/${encodeURIComponent(project.name)}" class="btn">Open</a>
            <button onclick="deleteProject('${escapeHtml(project.name)}')" class="btn btn-danger">Delete</button>
        </div>
    `;

    return card;
}

function createNewProjectCard() {
    const card = document.createElement('div');
    card.className = 'project-card new-project';

    card.innerHTML = `
        <a href="/projects/new" class="new-project-link">
            <div class="plus-icon">+</div>
            <p>New Project</p>
        </a>
    `;

    return card;
}

async function deleteProject(name) {
    if (!confirm(`Delete project "${name}"? This cannot be undone.`)) {
        return;
    }

    try {
        await api.deleteProject(name);
        await loadProjects(); // Refresh
    } catch (error) {
        alert('Failed to delete project: ' + error.message);
    }
}

function showError(container, message) {
    container.innerHTML = `
        <div class="error-message">
            <p>${escapeHtml(message)}</p>
            <button onclick="loadProjects()" class="btn">Retry</button>
        </div>
    `;
}

// Utility: Prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
```

**Success Criteria:**
- [ ] Dashboard loads projects from API
- [ ] Renders project cards dynamically
- [ ] Delete functionality works
- [ ] NO calculations (just renders API data)
- [ ] XSS protection (escapeHtml)

---

#### Task 3D.2: Processing View Logic
**File:** `web/ui/static/js/processing.js`

```javascript
/**
 * Processing UI Logic
 *
 * IMPORTANT: Just renders progress from WebSocket
 * - NO calculations (progress comes from API as 0.0-1.0)
 * - NO business logic
 * - Just updates DOM
 */

let websocket = null;
const projectName = document.getElementById('project-name').value;

document.addEventListener('DOMContentLoaded', () => {
    initializeWebSocket();
    setupEventListeners();
});

function initializeWebSocket() {
    websocket = api.connectWebSocket(projectName, handleProgressUpdate, handleWebSocketError);
}

function handleProgressUpdate(data) {
    // JUST RENDER - API sends ready-to-use data

    // Update progress bar (API sends 0.0-1.0, we just convert to %)
    updateProgressBar('stage-progress', data.progress);

    // Update text (API sends text, we just display it)
    if (data.stage) {
        setText('current-stage', data.stage);
    }

    if (data.message) {
        setText('stage-status', data.message);
        appendToLogs(data.message);
    }

    // Update status class
    updateStatusClass(data.status);
}

function updateProgressBar(elementId, progress) {
    // API sends 0.0-1.0, convert to percentage for CSS
    const percent = `${progress * 100}%`;
    document.getElementById(elementId).style.width = percent;
}

function setText(elementId, text) {
    document.getElementById(elementId).textContent = text;
}

function appendToLogs(message) {
    const logs = document.getElementById('log-output');
    logs.textContent += message + '\n';
    logs.scrollTop = logs.scrollHeight;
}

function updateStatusClass(status) {
    const statusElement = document.getElementById('stage-status');
    statusElement.className = `status-text status-${status}`;

    if (status === 'complete') {
        showCompletionMessage();
    } else if (status === 'failed') {
        showFailureMessage();
    }
}

function showCompletionMessage() {
    setText('stage-status', 'Processing complete!');
}

function showFailureMessage() {
    setText('stage-status', 'Processing failed');
}

function handleWebSocketError(error) {
    console.error('WebSocket error:', error);
    setText('stage-status', 'Connection lost. Retrying...');
}

function setupEventListeners() {
    document.getElementById('stop-btn').addEventListener('click', async () => {
        if (confirm('Stop processing? Progress will be lost.')) {
            try {
                await api.stopJob(projectName);
                window.location.href = `/projects/${projectName}`;
            } catch (error) {
                alert('Failed to stop job: ' + error.message);
            }
        }
    });

    document.getElementById('toggle-logs').addEventListener('click', () => {
        const logs = document.getElementById('log-output');
        const isHidden = logs.style.display === 'none';
        logs.style.display = isHidden ? 'block' : 'none';
        setText('toggle-logs', isHidden ? 'Hide' : 'Show');
    });
}
```

**Success Criteria:**
- [ ] WebSocket connects and receives updates
- [ ] Progress bar updates (no calculation - just renders)
- [ ] Logs append correctly
- [ ] Stop button works
- [ ] NO business logic (just presentation)

---

### Phase 3D Exit Criteria

- [ ] All UI logic implemented
- [ ] Zero business logic in frontend
- [ ] All API calls through api-client.js
- [ ] Proper error handling
- [ ] XSS protection where needed

---

## Phase 3E: Testing & Polish ⚪

**Goal:** End-to-end testing and UI refinements

### Deliverables
- Manual test plan
- Browser compatibility testing
- E2E tests (optional - Playwright/Cypress)
- Performance optimization

### Tasks

#### Task 3E.1: Manual Test Plan
**File:** `docs/UI-TEST-PLAN.md`

```markdown
# UI Test Plan

## Browser Testing

Test on:
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

## Functional Tests

### Dashboard
- [ ] Load dashboard - projects displayed
- [ ] Click "New Project" - navigate to form
- [ ] Click "Open" on project - navigate to project view
- [ ] Click "Delete" - confirm dialog, project deleted

### New Project
- [ ] Fill form with valid data - project created
- [ ] Fill form with invalid name - validation error
- [ ] Upload video - file uploaded, progress shown

### Processing
- [ ] Start job - WebSocket connects
- [ ] Progress updates in real-time
- [ ] Stop button works
- [ ] Logs toggle works

## Responsive Design
- [ ] Test on mobile (375px)
- [ ] Test on tablet (768px)
- [ ] Test on desktop (1920px)

## Performance
- [ ] Initial load < 2 seconds
- [ ] API calls complete < 500ms
- [ ] UI responsive (60fps)
```

**Success Criteria:**
- [ ] Test plan comprehensive
- [ ] All browsers tested
- [ ] No critical bugs

---

#### Task 3E.2: E2E Tests (Optional)
**File:** `web/tests/e2e/test_ui.spec.js` (Playwright)

```javascript
// Example E2E test
const { test, expect } = require('@playwright/test');

test('dashboard loads and displays projects', async ({ page }) => {
    await page.goto('http://localhost:5000/dashboard');

    // Wait for API call to complete
    await page.waitForSelector('.project-card');

    // Verify projects rendered
    const projects = await page.locator('.project-card').count();
    expect(projects).toBeGreaterThan(0);
});

test('create new project flow', async ({ page }) => {
    await page.goto('http://localhost:5000/projects/new');

    // Fill form
    await page.fill('#project-name', 'TestProject');
    await page.selectOption('#stages', ['ingest', 'colmap']);

    // Submit
    await page.click('#submit-btn');

    // Verify redirect to dashboard
    await expect(page).toHaveURL('/dashboard');
});
```

**Success Criteria:**
- [ ] E2E tests cover critical paths
- [ ] Tests run in CI/CD
- [ ] Fast execution (< 30s)

---

### Phase 3E Exit Criteria

- [ ] Manual testing complete
- [ ] All browsers supported
- [ ] Responsive design verified
- [ ] Performance acceptable
- [ ] No critical UI bugs
- [ ] Accessibility tested (keyboard navigation, screen readers)

---

## Phase 3F: Startup Scripts & Integration ⚪

**Goal:** One-command startup for end users

### Deliverables
- Startup script (opens browser)
- Shutdown handling
- Desktop shortcuts (optional)
- User documentation

### Tasks

#### Task 3F.1: Create Startup Script
**File:** `start-platform.sh`

```bash
#!/bin/bash
set -e

echo "=== VFX Ingest Platform ==="

# Check if already running
if docker ps | grep -q vfx-ingest; then
    echo "Platform already running!"
    URL="http://localhost:5000"
else
    # Start container
    echo "Starting platform..."
    docker-compose up -d

    # Wait for health check
    echo "Waiting for server..."
    for i in {1..30}; do
        if curl -s http://localhost:5000/api/system/health > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    URL="http://localhost:5000"
fi

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$URL"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    start "$URL"
fi

echo "✓ Platform running at $URL"
echo ""
echo "To stop: Use 'Shutdown' button in web interface"
```

**Success Criteria:**
- [ ] Script starts Docker
- [ ] Browser opens automatically
- [ ] Works on Linux, macOS, Windows (WSL)

---

### Phase 3F Exit Criteria

- [ ] One-command startup works
- [ ] Browser opens automatically
- [ ] Shutdown button functional
- [ ] User documentation complete

---

## Roadmap 3 Success Criteria

**Ready for production when:**

- [ ] All phases complete
- [ ] Web UI fully functional
- [ ] Zero business logic in frontend
- [ ] All API calls through abstraction layer
- [ ] Responsive design verified
- [ ] Browser compatibility tested
- [ ] Performance acceptable (< 2s load)
- [ ] Accessible (WCAG AA)
- [ ] User testing successful (3+ artists)
- [ ] Documentation complete
- [ ] One-click startup works

**UI Quality Checklist:**
- [ ] Clean, professional appearance
- [ ] Artist-friendly (no technical jargon)
- [ ] Responsive (mobile, tablet, desktop)
- [ ] Fast (< 2s initial load)
- [ ] Accessible (keyboard, screen readers)
- [ ] Error handling (graceful failures)
- [ ] Loading states (spinners, skeletons)
- [ ] Consistent styling (design system)

---

**Previous:** [Roadmap 2: API Backend](ROADMAP-2-API.md)
**Up:** [Atlas Overview](ATLAS.md)
