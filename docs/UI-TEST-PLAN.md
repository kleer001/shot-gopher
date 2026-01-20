# UI Test Plan - VFX Ingest Platform

**Version:** 1.0
**Last Updated:** 2026-01-20
**Status:** Active Testing Document

---

## Overview

This document outlines the comprehensive testing strategy for the VFX Ingest Platform web UI. All tests verify **presentation layer only** - the UI should make API calls and render responses without performing business logic.

---

## Browser Compatibility Testing

### Target Browsers

Test on the following browsers (latest stable versions):

- [ ] **Chrome** (latest) - Primary target
- [ ] **Firefox** (latest) - Secondary target
- [ ] **Safari** (latest) - macOS users
- [ ] **Edge** (latest) - Windows users

### Browser-Specific Tests

For each browser, verify:
- [ ] All layouts render correctly
- [ ] WebSocket connections work
- [ ] File uploads function properly
- [ ] CSS animations and transitions work
- [ ] JavaScript ES6 features supported
- [ ] Console has no errors

---

## Responsive Design Testing

### Device Sizes

Test at the following viewport widths:

- [ ] **Mobile** (375px) - iPhone SE
- [ ] **Mobile Large** (414px) - iPhone Plus
- [ ] **Tablet** (768px) - iPad
- [ ] **Tablet Large** (1024px) - iPad Pro
- [ ] **Desktop** (1440px) - Standard laptop
- [ ] **Desktop Large** (1920px) - Full HD monitor
- [ ] **Desktop XL** (2560px) - QHD monitor

### Responsive Checks

For each viewport:
- [ ] Navigation remains accessible
- [ ] Forms are usable
- [ ] Project cards layout properly
- [ ] Progress bars visible
- [ ] No horizontal scrolling
- [ ] Touch targets adequate (44x44px minimum)

---

## Functional Test Cases

### 1. Dashboard Tests

#### 1.1 Load Dashboard
**Steps:**
1. Navigate to `http://localhost:5000/dashboard`
2. Observe loading state
3. Wait for projects to load

**Expected:**
- [ ] Loading indicator shown initially
- [ ] Projects loaded from API within 2 seconds
- [ ] Project cards displayed in grid
- [ ] "New Project" card visible
- [ ] No JavaScript errors in console

#### 1.2 View Project Layouts
**Steps:**
1. Visit `/` (default layout)
2. Visit `/compact` (compact layout)
3. Visit `/dashboard` (dashboard layout)
4. Visit `/split` (split-screen layout)
5. Visit `/cards` (cards layout)

**Expected:**
- [ ] All layouts load without errors
- [ ] Each layout has distinct appearance
- [ ] Navigation works in all layouts
- [ ] API calls successful in all layouts

#### 1.3 Delete Project
**Steps:**
1. Click "Delete" on a project card
2. Confirm deletion in dialog
3. Observe project removal

**Expected:**
- [ ] Confirmation dialog appears
- [ ] Canceling preserves project
- [ ] Confirming deletes project
- [ ] Dashboard refreshes automatically
- [ ] Deleted project removed from grid

#### 1.4 Navigate to New Project
**Steps:**
1. Click "New Project" button
2. Observe navigation

**Expected:**
- [ ] Redirects to project creation form
- [ ] No console errors

---

### 2. Project Creation Tests

#### 2.1 Create Project (Valid Data)
**Steps:**
1. Fill in project name (e.g., "TestProject123")
2. Select stages (e.g., "ingest", "depth")
3. Click "Create Project"

**Expected:**
- [ ] Form validates locally
- [ ] API call made to `/api/projects`
- [ ] Success response handled
- [ ] Redirect to dashboard or project view
- [ ] New project appears in list

#### 2.2 Create Project (Invalid Name)
**Steps:**
1. Enter invalid project name (e.g., "Test/Project" with slash)
2. Click "Create Project"

**Expected:**
- [ ] Client-side validation error shown
- [ ] Form does not submit
- [ ] Error message descriptive
- [ ] User can correct and resubmit

#### 2.3 Upload Video
**Steps:**
1. Create or select project
2. Click file upload button
3. Select video file
4. Observe upload progress

**Expected:**
- [ ] File input accepts video formats only
- [ ] Upload progress shown (0-100%)
- [ ] Success message on completion
- [ ] Video metadata displayed
- [ ] Error handling for failed uploads

---

### 3. Processing View Tests

#### 3.1 Start Processing Job
**Steps:**
1. Create project with video
2. Select stages
3. Click "Start Processing"
4. Observe real-time updates

**Expected:**
- [ ] WebSocket connection established
- [ ] Progress bar initializes
- [ ] Current stage displayed
- [ ] Status messages update in real-time
- [ ] Progress advances (0% → 100%)

#### 3.2 Real-Time Progress Updates
**Steps:**
1. Start processing job
2. Observe WebSocket messages
3. Verify UI updates

**Expected:**
- [ ] Progress bar updates smoothly
- [ ] Current stage name updates
- [ ] Status text updates
- [ ] Logs append (if enabled)
- [ ] No UI freezing
- [ ] 60fps rendering

#### 3.3 Stop Processing Job
**Steps:**
1. Start processing job
2. Click "Stop Processing"
3. Confirm in dialog

**Expected:**
- [ ] Confirmation dialog appears
- [ ] API call to `/api/pipeline/projects/{name}/stop`
- [ ] WebSocket connection closed
- [ ] Redirect to project view
- [ ] Job status updated to "stopped"

#### 3.4 Toggle Logs
**Steps:**
1. Start processing job
2. Click "Show Logs"
3. Click "Hide Logs"

**Expected:**
- [ ] Logs section expands/collapses
- [ ] Button text toggles ("Show" ↔ "Hide")
- [ ] Logs auto-scroll to bottom
- [ ] No performance issues with large logs

#### 3.5 Completion Handling
**Steps:**
1. Wait for job to complete
2. Observe final state

**Expected:**
- [ ] Progress reaches 100%
- [ ] Status changes to "complete"
- [ ] Success message shown
- [ ] WebSocket closes gracefully
- [ ] "View Results" button enabled

#### 3.6 Failure Handling
**Steps:**
1. Trigger job failure (invalid config)
2. Observe error handling

**Expected:**
- [ ] Status changes to "failed"
- [ ] Error message displayed
- [ ] Logs show error details
- [ ] User can retry or go back

---

### 4. System Tests

#### 4.1 Shutdown System
**Steps:**
1. Click "Shutdown" button in navigation
2. Confirm shutdown

**Expected:**
- [ ] Confirmation dialog shown
- [ ] API call to `/api/system/shutdown`
- [ ] Server shuts down gracefully
- [ ] Clear message to user

#### 4.2 Health Check
**Steps:**
1. Visit `/health` endpoint
2. Check response

**Expected:**
- [ ] Returns `{"status": "ok"}`
- [ ] 200 HTTP status code

---

### 5. API Client Tests

#### 5.1 Error Handling
**Steps:**
1. Simulate API error (network disconnect)
2. Attempt operation

**Expected:**
- [ ] Error caught gracefully
- [ ] User-friendly error message shown
- [ ] No unhandled promise rejections
- [ ] Console shows descriptive error

#### 5.2 Request Timeout
**Steps:**
1. Simulate slow API response
2. Observe timeout behavior

**Expected:**
- [ ] Request times out appropriately
- [ ] Loading state shown during request
- [ ] Timeout error message displayed

---

## Accessibility Testing

### Keyboard Navigation

- [ ] Tab order logical (top to bottom, left to right)
- [ ] All interactive elements focusable
- [ ] Focus indicators visible (outline)
- [ ] Enter/Space activate buttons
- [ ] Escape closes dialogs
- [ ] Arrow keys navigate lists (optional)

### Screen Reader Testing

Test with:
- [ ] **NVDA** (Windows, free)
- [ ] **JAWS** (Windows, commercial)
- [ ] **VoiceOver** (macOS/iOS, built-in)

Verify:
- [ ] All images have alt text
- [ ] Form inputs have labels
- [ ] ARIA labels on custom controls
- [ ] Status updates announced
- [ ] Error messages announced
- [ ] Headings structure logical (h1 → h2 → h3)

### ARIA Attributes

Check for:
- [ ] `aria-label` on icon buttons
- [ ] `aria-live` on status regions
- [ ] `aria-busy` on loading states
- [ ] `role="status"` for status messages
- [ ] `role="alert"` for errors
- [ ] `role="progressbar"` on progress bars

### Color Contrast

Use [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/):
- [ ] Text meets WCAG AA (4.5:1 for normal, 3:1 for large)
- [ ] Interactive elements meet AA (3:1)
- [ ] Focus indicators meet AA (3:1)
- [ ] Status colors distinguishable (not just color)

---

## Performance Testing

### Load Time
- [ ] Initial page load < 2 seconds (on 3G)
- [ ] Time to Interactive < 3 seconds
- [ ] First Contentful Paint < 1.5 seconds

### API Response Time
- [ ] List projects < 500ms
- [ ] Get project < 500ms
- [ ] Create project < 1 second
- [ ] Upload video < 5 seconds (1GB file)

### JavaScript Performance
- [ ] No long tasks (> 50ms)
- [ ] 60fps rendering (no jank)
- [ ] WebSocket updates < 100ms latency
- [ ] Memory usage stable (no leaks)

### Bundle Size
- [ ] Total CSS < 50KB (uncompressed)
- [ ] Total JS < 200KB (uncompressed)
- [ ] No unused code (tree-shaking)

### Lighthouse Audit
Run Lighthouse in Chrome DevTools:
- [ ] Performance score > 90
- [ ] Accessibility score > 90
- [ ] Best Practices score > 90
- [ ] SEO score > 80 (optional)

---

## Security Testing

### Input Validation
- [ ] XSS protection (HTML escaping)
- [ ] SQL injection prevention (parameterized queries)
- [ ] Path traversal prevention
- [ ] File upload restrictions (type, size)

### HTTPS
- [ ] HTTPS enforced in production
- [ ] No mixed content warnings
- [ ] Valid SSL certificate

### CORS
- [ ] Appropriate CORS headers
- [ ] No overly permissive origins

---

## Error Scenarios

### Network Errors
- [ ] API server down → graceful error message
- [ ] WebSocket disconnect → auto-reconnect attempt
- [ ] Slow network → loading indicators shown
- [ ] Request timeout → clear error message

### User Errors
- [ ] Invalid form input → validation error
- [ ] Missing required field → clear indication
- [ ] Duplicate project name → descriptive error
- [ ] Upload unsupported file → format error

### System Errors
- [ ] Out of disk space → error message
- [ ] ComfyUI not available → warning shown
- [ ] Stage execution failure → error details in logs

---

## Cross-Browser Issues

### Known Issues to Check

1. **Safari:**
   - [ ] WebSocket support
   - [ ] ES6 module support
   - [ ] CSS Grid support

2. **Firefox:**
   - [ ] Fetch API support
   - [ ] FormData support
   - [ ] CSS Custom Properties

3. **Edge:**
   - [ ] Legacy Edge vs. Chromium Edge
   - [ ] IE11 compatibility (optional)

---

## Regression Testing

After any code change, verify:
- [ ] All layouts load without errors
- [ ] API client methods work
- [ ] WebSocket connections establish
- [ ] Forms validate and submit
- [ ] Progress updates display
- [ ] No console errors

---

## User Acceptance Testing

Recruit 3+ artists to test:

### Tasks
1. Create new project
2. Upload video file
3. Select and run stages
4. Monitor progress
5. View results
6. Delete project

### Feedback Questions
- Is the interface intuitive?
- Are error messages helpful?
- Is the progress tracking clear?
- Are there any confusing elements?
- What would improve the experience?

### Success Criteria
- [ ] 90% task completion rate
- [ ] < 2 support questions per user
- [ ] Positive feedback on usability
- [ ] No critical UX issues

---

## Automated Testing

### Unit Tests (JavaScript)
**File:** `web/tests/unit/test_ui_utils.js`

Test:
- [ ] `escapeHtml()` prevents XSS
- [ ] `formatTime()` formats correctly
- [ ] `formatBytes()` formats correctly
- [ ] DOM utility functions work

### Integration Tests (Python)
**File:** `web/tests/integration/test_ui_flows.py`

Test:
- [ ] Dashboard loads projects
- [ ] Create project flow works
- [ ] Upload video flow works
- [ ] Processing flow works
- [ ] Delete project works

### E2E Tests (Playwright/Selenium)
**File:** `web/tests/e2e/test_ui_e2e.py`

Test full user journeys:
- [ ] Complete project creation to results
- [ ] Multi-stage processing flow
- [ ] Error recovery scenarios

---

## Test Environment Setup

### Local Testing
```bash
# Start server
cd web
python -m uvicorn server:app --reload

# Run tests
pytest tests/
```

### Docker Testing
```bash
# Build and start
docker-compose up --build

# Run tests against container
pytest web/tests/integration/
```

---

## Bug Tracking

### Severity Levels
- **Critical:** App unusable, data loss, security issue
- **High:** Major feature broken, workaround exists
- **Medium:** Minor feature broken, cosmetic issue
- **Low:** Enhancement, nice-to-have

### Bug Report Template
```markdown
**Title:** Brief description
**Severity:** Critical/High/Medium/Low
**Browser:** Chrome 120 / Firefox 121 / etc.
**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected:** What should happen
**Actual:** What actually happens
**Screenshots:** Attach if applicable
**Console Logs:** Include errors
```

---

## Test Results Tracking

### Test Run Log

| Date | Tester | Browser | Pass | Fail | Notes |
|------|--------|---------|------|------|-------|
| 2026-01-20 | AI | Chrome | 0 | 0 | Initial |

---

## Checklist Summary

### Phase 3E Completion Criteria

- [ ] All browsers tested (Chrome, Firefox, Safari, Edge)
- [ ] All responsive breakpoints tested
- [ ] All functional tests pass
- [ ] Accessibility score > 90
- [ ] Performance score > 90
- [ ] No critical bugs
- [ ] User testing complete (3+ users)

---

**Next Steps:**
1. Execute all test cases
2. Document bugs found
3. Fix critical and high-severity bugs
4. Re-test after fixes
5. Get user feedback
6. Iterate and improve
