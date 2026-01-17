# 🗺️ VFX Ingest Platform - Containerization Atlas

## Overview

This atlas documents the strategic transition of the VFX Ingest Platform from a local conda-based installation to a containerized, web-accessible system optimized for artist-friendly deployment.

## Purpose

The current installation requires technical expertise:
- Conda environment setup
- COLMAP compilation/installation challenges
- Multiple model downloads with authentication
- Command-line interface

The atlas guides the transition to:
- Docker-based deployment (eliminates installation complexity)
- Web-based interface (artist-friendly GUI)
- Robust, repeatable environments

## Atlas Structure

This atlas contains two sequential roadmaps:

### 📋 [Roadmap 1: Docker Migration](ROADMAP-1-DOCKER.md)
**Status:** 🟡 In Progress
**Goal:** Replicate current CLI functionality in Docker containers

Migrate existing Python CLI workflow into Docker containers while maintaining exact feature parity. Users still interact via command-line but benefit from containerized dependencies.

**Key Outcomes:**
- Dockerfile + docker-compose configuration
- Container-aware code modifications
- Model persistence via volume mounts
- Validated feature parity with local installation

**Target Timeline:** 2-3 weeks

---

### 🌐 [Roadmap 2: Web Interface](ROADMAP-2-WEB.md)
**Status:** ⚪ Not Started
**Goal:** Add browser-based GUI layer over containerized backend

Build a FastAPI web application that wraps the containerized pipeline, providing an intuitive web interface for non-technical artists.

**Key Outcomes:**
- FastAPI web server with dashboard
- Project management UI
- Real-time progress tracking
- One-click startup/shutdown

**Target Timeline:** 3-4 weeks

---

## Dependencies

```
Roadmap 1 (Docker)
    ↓
    Roadmap 2 (Web)
```

Roadmap 2 **depends on** Roadmap 1 completion. Docker migration must be validated before building web layer.

## Success Criteria

### Roadmap 1 Complete When:
- [ ] `docker-compose up` starts all services
- [ ] All existing pipeline stages work identically
- [ ] Models persist between container restarts
- [ ] Output files accessible from host
- [ ] COLMAP works without host installation
- [ ] Performance comparable to local installation

### Roadmap 2 Complete When:
- [ ] One-command startup opens browser interface
- [ ] Artists can create/manage projects via GUI
- [ ] Real-time progress visible in browser
- [ ] Graceful shutdown with job protection
- [ ] All Roadmap 1 functionality accessible via web

## Current State

**Branch:** `claude/containerize-colmap-pipeline-vAOqu`

**Existing Architecture:**
- Python 3.10+ conda environment
- ComfyUI with custom nodes (SAM3, Depth Anything, MatAnyone, ProPainter)
- COLMAP for camera tracking
- FFmpeg for video processing
- Multiple ML models (~15-20GB total)
- Project-based directory structure

**Pain Points Being Addressed:**
- COLMAP installation difficulties across platforms
- Complex conda environment setup
- Command-line interface barrier for artists
- Dependency version conflicts

## Navigation

- **Start Here:** [Roadmap 1: Docker Migration](ROADMAP-1-DOCKER.md)
- **Future:** [Roadmap 2: Web Interface](ROADMAP-2-WEB.md)

## Maintenance

These documents are living specifications and should be updated as:
- Phases complete (mark with ✅)
- Blockers encountered (document in phase notes)
- Requirements change (update success criteria)
- New dependencies discovered (add to relevant roadmap)

---

**Last Updated:** 2026-01-17
**Current Phase:** Roadmap 1, Phase 1A
**Next Milestone:** Working Dockerfile with all dependencies
