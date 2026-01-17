/**
 * VFX Pipeline Web Interface
 * Client-side application logic
 */

// State management
const state = {
    currentSection: 'upload',
    projectId: null,
    projectDir: null,
    projectName: null,
    videoInfo: null,
    stages: [],
    ws: null,
    startTime: null,
};

// DOM Elements
const elements = {
    // Upload & Config
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    browseBtn: document.getElementById('browse-btn'),
    uploadProgress: document.getElementById('upload-progress'),
    uploadFilename: document.getElementById('upload-filename'),
    uploadProgressFill: document.getElementById('upload-progress-fill'),
    uploadPercentText: document.getElementById('upload-percent-text'),
    videoInfo: document.getElementById('video-info'),
    videoName: document.getElementById('video-name'),
    videoResolution: document.getElementById('video-resolution'),
    videoFrames: document.getElementById('video-frames'),
    videoFps: document.getElementById('video-fps'),
    configForm: document.getElementById('config-form'),
    rotoPromptInput: document.getElementById('roto-prompt'),
    rotoPromptWrapper: document.getElementById('roto-prompt-wrapper'),
    stageRoto: document.getElementById('stage-roto'),
    skipExisting: document.getElementById('skip-existing'),
    presetButtons: document.querySelectorAll('.preset-btn'),
    timeEstimate: document.getElementById('time-estimate'),

    // System status
    systemStatus: document.getElementById('system-status'),

    // Projects
    projectsList: document.getElementById('projects-list'),

    // Processing Panel
    processingPanel: document.getElementById('processing-panel'),
    processingProjectName: document.getElementById('processing-project-name'),
    cancelProcessingBtn: document.getElementById('cancel-processing-btn'),
    currentStageLabel: document.getElementById('current-stage-label'),
    currentStageName: document.getElementById('current-stage-name'),
    processingProgressFill: document.getElementById('processing-progress-fill'),
    progressPercent: document.getElementById('progress-percent'),
    progressFrames: document.getElementById('progress-frames'),
    elapsedTime: document.getElementById('elapsed-time'),
    remainingTime: document.getElementById('remaining-time'),
    stagesListProgress: document.getElementById('stages-list-progress'),
    clearLogsBtn: document.getElementById('clear-logs-btn'),
    logOutput: document.getElementById('log-output'),

    // Error Toast
    errorToast: document.getElementById('error-toast'),
    errorMessage: document.getElementById('error-message'),
    errorClose: document.getElementById('error-close'),

    // Classic template only elements
    logContainer: document.getElementById('log-container'),
    toggleLogBtn: document.getElementById('toggle-log-btn'),
    outputsGrid: document.getElementById('outputs-grid'),
    completeProjectName: document.getElementById('complete-project-name'),
    cancelConfigBtn: document.getElementById('cancel-config-btn'),
    openFolderBtn: document.getElementById('open-folder-btn'),
    runAgainBtn: document.getElementById('run-again-btn'),
    newProjectBtn: document.getElementById('new-project-btn'),
    errorBackBtn: document.getElementById('error-back-btn'),

    // SVG progress indicators (Dashboard and Split templates)
    progressRing: document.getElementById('progress-ring'),
    progressCircle: document.getElementById('progress-circle'),
};

// Presets configuration
const presets = {
    quick: ['depth', 'roto'],
    full: ['depth', 'roto', 'cleanplate'],
    all: ['depth', 'roto', 'matanyone', 'cleanplate', 'colmap', 'gsir', 'mocap'],
};

// Stage display names
const stageNames = {
    ingest: 'Ingest',
    depth: 'Depth Maps',
    roto: 'Segmentation',
    matanyone: 'Refine Mattes',
    cleanplate: 'Clean Plate',
    colmap: 'Camera Solve',
    gsir: 'Materials',
    mocap: 'Motion Capture',
    camera: 'Camera Export',
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkSystemStatus();
    loadProjects();
});

function setupEventListeners() {
    // File upload
    elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('click', (e) => {
        if (e.target === elements.dropZone || e.target.closest('.drop-zone-content')) {
            elements.fileInput.click();
        }
    });
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);

    // Configure
    elements.configForm.addEventListener('submit', handleStartProcessing);
    elements.stageRoto.addEventListener('change', toggleRotoPrompt);

    // Stage dependency handling
    document.querySelectorAll('input[name="stage"]').forEach(checkbox => {
        checkbox.addEventListener('change', handleStageChange);
    });

    // Presets
    elements.presetButtons.forEach(btn => {
        btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });

    // Processing
    elements.cancelProcessingBtn.addEventListener('click', handleCancelProcessing);
    if (elements.clearLogsBtn) {
        elements.clearLogsBtn.addEventListener('click', () => {
            elements.logOutput.innerHTML = '';
        });
    }
    if (elements.toggleLogBtn) {
        elements.toggleLogBtn.addEventListener('click', toggleLog);
    }

    // Classic template buttons
    if (elements.cancelConfigBtn) {
        elements.cancelConfigBtn.addEventListener('click', resetToUpload);
    }
    if (elements.openFolderBtn) {
        elements.openFolderBtn.addEventListener('click', handleOpenFolder);
    }
    if (elements.runAgainBtn) {
        elements.runAgainBtn.addEventListener('click', resetToUpload);
    }
    if (elements.newProjectBtn) {
        elements.newProjectBtn.addEventListener('click', resetToUpload);
    }
    if (elements.errorBackBtn) {
        elements.errorBackBtn.addEventListener('click', resetToUpload);
    }

    // Error Toast
    if (elements.errorClose) {
        elements.errorClose.addEventListener('click', () => {
            elements.errorToast.classList.add('hidden');
        });
    }
}

// UI State Management
function showProcessingPanel() {
    elements.processingPanel.classList.remove('hidden');
}

function hideProcessingPanel() {
    elements.processingPanel.classList.add('hidden');
}

function showError(message, duration = 5000) {
    elements.errorMessage.textContent = message;
    elements.errorToast.classList.remove('hidden');

    if (duration > 0) {
        setTimeout(() => {
            elements.errorToast.classList.add('hidden');
        }, duration);
    }
}

function resetUploadForm() {
    elements.dropZone.classList.remove('hidden');
    elements.uploadProgress.classList.add('hidden');
    elements.videoInfo.classList.add('hidden');
    elements.configForm.classList.add('hidden');
    state.videoInfo = null;
    state.projectId = null;
}

// Legacy compatibility stub
function showSection(name) {
    // Single-page layout doesn't use sections
    // Keeping for backward compatibility with existing code
    console.log(`showSection called with: ${name}`);
}

// System status
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/system/status');
        const data = await response.json();

        if (data.comfyui) {
            elements.systemStatus.classList.add('online');
            elements.systemStatus.classList.remove('offline');
            elements.systemStatus.querySelector('.status-text').textContent = 'ONLINE';
        } else {
            elements.systemStatus.classList.add('offline');
            elements.systemStatus.classList.remove('online');
            elements.systemStatus.querySelector('.status-text').textContent = 'OFFLINE';
        }
    } catch (error) {
        console.error('Failed to check system status:', error);
    }
}

// Load recent projects
async function loadProjects() {
    try {
        const response = await fetch('/api/projects');
        const data = await response.json();

        if (data.projects && data.projects.length > 0) {
            elements.projectsList.innerHTML = data.projects.slice(0, 5).map(proj => `
                <div class="project-item" data-id="${proj.project_id}" data-dir="${proj.project_dir}">
                    <span class="project-name">${proj.name || proj.project_id}</span>
                    <span class="project-status ${proj.status}">${proj.status}</span>
                </div>
            `).join('');

            // Add click handlers
            elements.projectsList.querySelectorAll('.project-item').forEach(item => {
                item.addEventListener('click', () => {
                    state.projectId = item.dataset.id;
                    state.projectDir = item.dataset.dir;
                    loadProjectOutputs(item.dataset.id);
                });
            });
        } else {
            elements.projectsList.innerHTML = '<p class="no-projects">(none yet)</p>';
        }
    } catch (error) {
        console.error('Failed to load projects:', error);
        elements.projectsList.innerHTML = '<p class="no-projects">Failed to load</p>';
    }
}

// File handling
function handleDragOver(e) {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

async function uploadFile(file) {
    // Validate file type
    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mxf'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!validExtensions.includes(ext)) {
        showError(`Unsupported file type: ${ext}`);
        return;
    }

    // Show progress
    elements.uploadProgress.classList.remove('hidden');
    elements.uploadFilename.textContent = file.name;
    elements.uploadProgressFill.style.width = '0%';
    elements.uploadPercentText.textContent = '0';

    // Create form data
    const formData = new FormData();
    formData.append('file', file);

    try {
        const xhr = new XMLHttpRequest();

        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const percent = Math.round((e.loaded / e.total) * 100);
                elements.uploadProgressFill.style.width = `${percent}%`;
                elements.uploadPercentText.textContent = percent;
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                handleUploadSuccess(data, file.name);
            } else {
                const error = JSON.parse(xhr.responseText);
                showError(error.detail || 'Upload failed');
            }
        });

        xhr.addEventListener('error', () => {
            showError('Upload failed - network error');
        });

        xhr.open('POST', '/api/upload');
        xhr.send(formData);
    } catch (error) {
        showError('Upload failed: ' + error.message);
    }
}

function handleUploadSuccess(data, filename) {
    state.projectId = data.project_id;
    state.projectDir = data.project_dir;
    state.projectName = data.name;
    state.videoInfo = data.video_info;

    // Update video info display
    if (data.video_info) {
        const info = data.video_info;
        elements.videoName.textContent = filename;
        elements.videoResolution.textContent = `${info.resolution[0]}x${info.resolution[1]}`;
        elements.videoFrames.textContent = info.frame_count;
        elements.videoFps.textContent = info.fps.toFixed(2);
        elements.videoInfo.classList.remove('hidden');
    }

    // Hide upload progress, show config form
    elements.uploadProgress.classList.add('hidden');
    elements.configForm.classList.remove('hidden');

    // Update time estimate with video info
    updateTimeEstimate();
}

// Configure form
function toggleRotoPrompt() {
    const isChecked = elements.stageRoto.checked;
    elements.rotoPromptWrapper.style.display = isChecked ? 'flex' : 'none';
}

function handleStageChange(e) {
    const checkbox = e.target;
    const stage = checkbox.value;

    // Handle COLMAP dependencies
    if (stage === 'colmap') {
        const gsirCheckbox = document.querySelector('input[value="gsir"]');
        const mocapCheckbox = document.querySelector('input[value="mocap"]');

        if (checkbox.checked) {
            gsirCheckbox.disabled = false;
            mocapCheckbox.disabled = false;
        } else {
            gsirCheckbox.disabled = true;
            gsirCheckbox.checked = false;
            mocapCheckbox.disabled = true;
            mocapCheckbox.checked = false;
        }
    }

    // Update preset selection
    updatePresetSelection();

    // Update time estimate
    updateTimeEstimate();
}

// Time estimates per frame (in seconds) - rough averages
const stageTimePerFrame = {
    ingest: 0.05,       // FFmpeg - fast
    depth: 0.5,         // ComfyUI depth estimation
    roto: 0.8,          // ComfyUI segmentation
    matanyone: 1.0,     // ComfyUI matte refinement
    cleanplate: 0.6,    // ComfyUI inpainting
    colmap: 2.0,        // COLMAP reconstruction (per frame average)
    gsir: 0.1,          // GS-IR training (amortized per frame)
    mocap: 1.5,         // Motion capture per frame
    camera: 0.01,       // Camera export - fast
};

function updateTimeEstimate() {
    if (!state.videoInfo || !state.videoInfo.frame_count) {
        if (elements.timeEstimate) {
            elements.timeEstimate.textContent = 'Unknown';
        }
        return;
    }

    const frameCount = state.videoInfo.frame_count;
    const selectedStages = Array.from(document.querySelectorAll('input[name="stage"]:checked'))
        .map(cb => cb.value);

    // Always include ingest
    let stages = ['ingest', ...selectedStages];

    // Add camera if colmap selected
    if (selectedStages.includes('colmap') && !stages.includes('camera')) {
        stages.push('camera');
    }

    // Calculate total time
    let totalSeconds = 0;
    stages.forEach(stage => {
        totalSeconds += (stageTimePerFrame[stage] || 0) * frameCount;
    });

    // Format the estimate
    let estimate;
    if (totalSeconds < 60) {
        estimate = `~${Math.round(totalSeconds)} seconds`;
    } else if (totalSeconds < 3600) {
        const mins = Math.round(totalSeconds / 60);
        estimate = `~${mins} minute${mins > 1 ? 's' : ''}`;
    } else {
        const hours = Math.floor(totalSeconds / 3600);
        const mins = Math.round((totalSeconds % 3600) / 60);
        estimate = `~${hours}h ${mins}m`;
    }

    if (elements.timeEstimate) {
        elements.timeEstimate.textContent = estimate;
    }
}

function applyPreset(presetName) {
    const stages = presets[presetName];
    if (!stages) return;

    // Clear all checkboxes
    document.querySelectorAll('input[name="stage"]').forEach(cb => {
        cb.checked = stages.includes(cb.value);

        // Enable COLMAP-dependent stages if COLMAP is selected
        if (cb.value === 'gsir' || cb.value === 'mocap') {
            cb.disabled = !stages.includes('colmap');
        }
    });

    // Update preset button selection
    elements.presetButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.preset === presetName);
    });

    // Update roto prompt visibility
    toggleRotoPrompt();

    // Update time estimate
    updateTimeEstimate();
}

function updatePresetSelection() {
    const selectedStages = Array.from(document.querySelectorAll('input[name="stage"]:checked'))
        .map(cb => cb.value);

    // Find matching preset
    let matchedPreset = null;
    for (const [name, stages] of Object.entries(presets)) {
        if (stages.length === selectedStages.length &&
            stages.every(s => selectedStages.includes(s))) {
            matchedPreset = name;
            break;
        }
    }

    // Update buttons
    elements.presetButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.preset === matchedPreset);
    });
}

async function handleStartProcessing(e) {
    e.preventDefault();

    // Get selected stages
    const selectedStages = Array.from(document.querySelectorAll('input[name="stage"]:checked'))
        .map(cb => cb.value);

    if (selectedStages.length === 0) {
        showError('Please select at least one processing stage');
        return;
    }

    // Always include ingest
    if (!selectedStages.includes('ingest')) {
        selectedStages.unshift('ingest');
    }

    // Always include camera export if colmap selected
    if (selectedStages.includes('colmap') && !selectedStages.includes('camera')) {
        selectedStages.push('camera');
    }

    state.stages = selectedStages;

    const config = {
        stages: selectedStages,
        roto_prompt: elements.rotoPromptInput.value || 'person',
        skip_existing: elements.skipExisting.checked,
    };

    try {
        const response = await fetch(`/api/projects/${state.projectId}/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start processing');
        }

        // Setup processing UI
        setupProcessingUI();

        // Connect WebSocket
        connectWebSocket();

        // Show processing panel
        showProcessingPanel();

        // Start timer
        state.startTime = Date.now();
        updateElapsedTime();
    } catch (error) {
        showError(error.message);
    }
}

// Processing UI
function setupProcessingUI() {
    elements.processingProjectName.textContent = state.projectName;

    // Build stages list
    elements.stagesListProgress.innerHTML = state.stages.map((stage, index) => `
        <div class="stage-item pending" data-stage="${stage}">
            <span class="stage-icon">○</span>
            <span>${stageNames[stage] || stage}</span>
            <span class="stage-status">pending</span>
        </div>
    `).join('');

    // Reset progress
    elements.processingProgressFill.style.width = '0%';
    elements.progressPercent.textContent = '0%';
    elements.progressFrames.textContent = '0 / 0';
    elements.currentStageLabel.textContent = `STAGE 1/${state.stages.length}`;
    elements.currentStageName.textContent = (stageNames[state.stages[0]] || state.stages[0]).toUpperCase();

    // Reset SVG progress rings
    updateSVGProgress(0);

    // Clear log
    elements.logOutput.innerHTML = '';
}

function connectWebSocket() {
    if (state.ws) {
        state.ws.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${state.projectId}`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleProgressUpdate(data);
    };

    state.ws.onclose = () => {
        // Reconnect if still processing (projectId is set and processing panel is visible)
        if (state.projectId && elements.processingPanel && !elements.processingPanel.classList.contains('hidden')) {
            setTimeout(connectWebSocket, 2000);
        }
    };

    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    // Setup ping interval
    const pingInterval = setInterval(() => {
        if (state.ws && state.ws.readyState === WebSocket.OPEN) {
            state.ws.send(JSON.stringify({ type: 'ping' }));
        } else {
            clearInterval(pingInterval);
        }
    }, 25000);
}

function handleProgressUpdate(data) {
    if (data.type === 'ping' || data.type === 'pong') return;

    // Check for completion
    if (data.status === 'completed') {
        handleProcessingComplete();
        return;
    }

    if (data.status === 'failed') {
        showError(data.error || 'Processing failed');
        return;
    }

    // Update progress bar
    if (data.progress !== undefined && data.progress > 0) {
        // Determinate mode - show actual progress
        elements.processingProgressFill.parentElement.classList.remove('indeterminate');
        const percent = Math.round(data.progress * 100);
        elements.processingProgressFill.style.width = `${percent}%`;
        elements.progressPercent.textContent = `${percent}%`;

        // Update SVG progress rings if they exist (Dashboard/Split templates)
        updateSVGProgress(percent);
    } else if (data.progress === undefined && data.stage) {
        // Indeterminate mode - no progress info yet for this stage
        elements.processingProgressFill.parentElement.classList.add('indeterminate');
        elements.progressPercent.textContent = 'Processing...';
    }

    // Update frame count
    if (data.frame !== undefined && data.total_frames !== undefined) {
        elements.progressFrames.textContent = `Frame ${data.frame} / ${data.total_frames}`;
    }

    // Update current stage
    if (data.stage) {
        const stageIndex = data.stage_index !== undefined ? data.stage_index : state.stages.indexOf(data.stage);
        elements.currentStageLabel.textContent = `STAGE ${stageIndex + 1}/${data.total_stages || state.stages.length}`;
        elements.currentStageName.textContent = (stageNames[data.stage] || data.stage).toUpperCase();

        // Update stages list
        updateStagesList(data.stage, stageIndex);
    }

    // Update log
    if (data.message) {
        appendLog(data.message);
    }
}

function updateStagesList(currentStage, currentIndex) {
    const stageItems = elements.stagesListProgress.querySelectorAll('.stage-item');

    stageItems.forEach((item, index) => {
        const icon = item.querySelector('.stage-icon');
        const status = item.querySelector('.stage-status');

        if (index < currentIndex) {
            // Completed
            icon.className = 'stage-icon completed';
            icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20 6 9 17 4 12"></polyline>
            </svg>`;
            status.textContent = 'completed';
        } else if (index === currentIndex) {
            // Processing
            icon.className = 'stage-icon processing';
            icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="12 6 12 12 16 14"></polyline>
            </svg>`;
            status.textContent = 'processing...';
        } else {
            // Pending
            icon.className = 'stage-icon pending';
            icon.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
            </svg>`;
            status.textContent = 'pending';
        }
    });
}

function updateSVGProgress(percent) {
    // Update Dashboard template progress ring (r=54)
    if (elements.progressRing) {
        const radius = 54;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference * (1 - percent / 100);
        elements.progressRing.style.strokeDasharray = circumference;
        elements.progressRing.style.strokeDashoffset = offset;
    }

    // Update Split template progress circle (r=90)
    if (elements.progressCircle) {
        const radius = 90;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference * (1 - percent / 100);
        elements.progressCircle.style.strokeDasharray = circumference;
        elements.progressCircle.style.strokeDashoffset = offset;
    }
}

function appendLog(message) {
    elements.logOutput.textContent += message + '\n';
    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
}

function toggleLog() {
    if (!elements.logContainer || !elements.toggleLogBtn) return;
    const isHidden = elements.logContainer.classList.toggle('hidden');
    elements.toggleLogBtn.innerHTML = isHidden ?
        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"></polyline>
        </svg> Show Log` :
        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="18 15 12 9 6 15"></polyline>
        </svg> Hide Log`;
}

function updateElapsedTime() {
    if (!state.startTime || !elements.processingPanel || elements.processingPanel.classList.contains('hidden')) return;

    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    elements.elapsedTime.textContent = `Elapsed: ${formatTime(elapsed)}`;

    requestAnimationFrame(() => setTimeout(updateElapsedTime, 1000));
}

function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const remainMins = mins % 60;
    return `${hours}h ${remainMins}m`;
}

async function handleCancelProcessing() {
    if (!confirm('Are you sure you want to cancel processing?')) return;

    try {
        await fetch(`/api/projects/${state.projectId}/stop`, { method: 'POST' });
        showSection('upload');
        loadProjects();
    } catch (error) {
        console.error('Failed to cancel:', error);
    }
}

// Complete
async function handleProcessingComplete() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    // Calculate processing time
    if (state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        console.log(`Processing complete in ${formatTime(elapsed)}`);
    }

    // Hide processing panel after a moment
    setTimeout(() => {
        hideProcessingPanel();
    }, 2000);

    // Reload projects list
    loadProjects();

    // Reset upload form for new project
    resetUploadForm();
}

async function loadProjectOutputs(projectId) {
    try {
        const response = await fetch(`/api/projects/${projectId}/outputs`);
        const data = await response.json();

        state.projectDir = data.project_dir;

        // Build output cards (only if outputs grid exists - classic template only)
        if (!elements.outputsGrid) {
            console.log('Outputs grid not available in this template');
            return;
        }

        const outputs = data.outputs;
        if (Object.keys(outputs).length === 0) {
            elements.outputsGrid.innerHTML = '<p>No outputs found</p>';
            return;
        }

        elements.outputsGrid.innerHTML = Object.entries(outputs).map(([name, info]) => `
            <div class="output-card">
                <div class="output-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
                        <polyline points="13 2 13 9 20 9"></polyline>
                    </svg>
                </div>
                <div class="output-name">${name}</div>
                <div class="output-count">${info.total_files} files</div>
            </div>
        `).join('');

        if (elements.completeProjectName) {
            elements.completeProjectName.textContent = projectId;
        }
        showSection('complete');
    } catch (error) {
        console.error('Failed to load outputs:', error);
    }
}

async function handleOpenFolder() {
    try {
        await fetch(`/api/projects/${state.projectId}/open-folder`, { method: 'POST' });
    } catch (error) {
        console.error('Failed to open folder:', error);
    }
}

function resetToUpload() {
    state.projectId = null;
    state.projectDir = null;
    state.projectName = null;
    state.videoInfo = null;
    state.stages = [];

    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }

    // Reset form
    elements.configForm.reset();
    elements.uploadProgress.classList.add('hidden');
    elements.fileInput.value = '';

    // Apply default preset
    applyPreset('full');

    // Reload projects
    loadProjects();

    showSection('upload');
}

// Initialize with full preset
applyPreset('full');
