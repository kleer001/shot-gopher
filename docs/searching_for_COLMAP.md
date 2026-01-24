# COLMAP Docker Integration: Troubleshooting Log

This document tracks all attempts to get COLMAP working in our Docker container with CUDA support.

## Environment

### Host System
- **OS**: Linux (kernel 4.4.0)
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **Docker**: With NVIDIA Container Toolkit

### Docker Container Target
- **Base Image**: `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`
- **Ubuntu**: 22.04 LTS (Jammy Jellyfish)
- **CUDA**: 12.1
- **cuDNN**: 8
- **GCC**: 10 (required for CUDA 12.1 compatibility on Ubuntu 22.04)
- **Python**: 3.10

### COLMAP Requirements
- **Goal**: GPU-accelerated SIFT feature extraction and matching
- **CUDA Architectures**: 75 (RTX 20xx/T4), 80 (A100), 86 (RTX 30xx), 89 (RTX 40xx), 90 (H100)
- **Features Needed**: Feature extraction, matching, sparse reconstruction, mesh generation (CGAL)

---

## Attempts (Chronological)

### Attempt 1: Copy Binary from Official Docker Image

**Commit**: `05281aa`, `ed51741`

**Approach**:
```dockerfile
FROM colmap/colmap:latest AS colmap-source
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS base

COPY --from=colmap-source /usr/local/bin/colmap /usr/local/bin/colmap
COPY --from=colmap-source /usr/local/lib/libcolmap* /usr/local/lib/
COPY --from=colmap-source /usr/lib/x86_64-linux-gnu/libfreeimage* /usr/lib/x86_64-linux-gnu/
```

**Result**: ❌ FAILED - "COLMAP not found"

**Why it failed**:
- Official `colmap/colmap:latest` is built on **Ubuntu 24.04 + CUDA 12.9**
- Our base is **Ubuntu 22.04 + CUDA 12.1**
- Library ABI mismatch (libboost 1.83 vs 1.74, glibc 2.39 vs 2.35, etc.)
- Binary expects Ubuntu 24.04 libraries but finds Ubuntu 22.04 versions

---

### Attempt 2: Install COLMAP via apt

**Commit**: `68d6fcd`

**Approach**:
```dockerfile
RUN apt-get install -y colmap
```

**Result**: ❌ FAILED - No CUDA support

**Why it failed**:
- Ubuntu's apt COLMAP package is **CPU-only**
- No GPU-accelerated SIFT extraction
- Per [COLMAP docs](https://colmap.github.io/install.html): "The COLMAP packages in the default repositories for Linux do not come with CUDA support"

---

### Attempt 3: Build from Source (COLMAP 3.9.1)

**Commit**: `8a71a89`, `9e79385`, `024a8d2`, `3328b8f`, `91790c7`, `4c484e6`

**Approach**:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS colmap-builder

ENV CC=/usr/bin/gcc-10
ENV CXX=/usr/bin/g++-10

RUN git clone --branch 3.9.1 --depth 1 https://github.com/colmap/colmap.git
RUN cmake .. -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" ...
```

**Build Issues Fixed**:
1. Missing `libboost-filesystem-dev` → Added
2. Missing CGAL → Added `libcgal-dev` (or `-DCGAL_ENABLED=OFF`)
3. Wrong runtime package `libcgal13` → Removed (CGAL is header-only on Ubuntu 22.04)
4. Added complete dependency list from official docs

**Result**: ❌ FAILED at runtime - "Failed to read image file format"

**Why it failed**:
- FreeImage initialization bug ([COLMAP issue #1548](https://github.com/colmap/colmap/issues/1548))
- System `libfreeimage3` doesn't call `FreeImage_Initialise()` properly
- `FreeImage_GetFileType()` returns `FIF_UNKNOWN` for all images
- All 125 frames fail with "Failed to read image file format"

---

### Attempt 4: Build from Source (COLMAP main branch)

**Commit**: `bd1c23d`

**Approach**:
```dockerfile
RUN git clone --depth 1 https://github.com/colmap/colmap.git /colmap-src
# No version tag = main branch
```

**Result**: ❌ Not recommended

**Why we moved away**:
- Main branch is unstable
- Non-reproducible builds
- Risk of breaking API changes
- PR #1549 (in main) only fixes **statically linked** FreeImage, not dynamic

---

### Attempt 5: Build from Source (COLMAP 3.10)

**Commit**: `22a4962`

**Approach**:
```dockerfile
ARG COLMAP_VERSION=3.10
RUN git clone --branch ${COLMAP_VERSION} --depth 1 https://github.com/colmap/colmap.git
```

**Rationale**:
- COLMAP 3.10 has PR #2332 "Fully encapsulate freeimage in bitmap library"
- This should fix dynamic linking issues
- Still C++14 (3.11+ requires C++17)

**Result**: ❌ FAILED - Same "Failed to read image file format" error

**Why it failed**:
- PR #2332 encapsulates FreeImage but still uses the system library
- Ubuntu 22.04's `libfreeimage3` has fundamental initialization issues
- The encapsulation doesn't fix the underlying library bug

---

### Attempt 6: Use OpenImageIO Instead of FreeImage

**Commit**: `4c6761f`

**Approach**:
```dockerfile
# Build stage
RUN apt-get install -y libopenimageio-dev  # instead of libfreeimage-dev

# Runtime stage
RUN apt-get install -y libopenimageio2.4   # instead of libfreeimage3
```

**Rationale**:
- Official COLMAP Docker image uses OpenImageIO, not FreeImage
- OpenImageIO is industry standard (used in OpenEXR, major VFX tools)
- More robust and actively maintained

**Result**: ❓ UNTESTED (current state)

---

## Current Error

```
E0122 01:58:03.312772   244 feature_extraction.cc:266] Failed to read image file format.
```

For ALL images. Both GPU and CPU extraction fail identically.

---

## Things NOT Yet Tried

### 1. Force COLMAP to Use OpenImageIO at Build Time

COLMAP's CMake might still prefer FreeImage if both are available. Try:

```dockerfile
RUN cmake .. \
    -DFREEIMAGE_ENABLED=OFF \
    -DOPENIMAGEIO_ENABLED=ON \
    ...
```

Or ensure FreeImage is NOT installed:
```dockerfile
# Don't install libfreeimage-dev at all
# Only install libopenimageio-dev
```

### 2. Build COLMAP with Bundled/Static FreeImage

```dockerfile
RUN cmake .. \
    -DFETCH_FREEIMAGE=ON \
    ...
```

This would download and statically link FreeImage, avoiding the system library issues.

### 3. Use COLMAP 3.11+ with C++17

COLMAP 3.11+ drops FreeImage in favor of std::filesystem and may have better image handling:

```dockerfile
ARG COLMAP_VERSION=3.11.0
# But requires GCC 11+ and C++17
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
```

**Risk**: May conflict with CUDA 12.1's GCC requirements.

### 4. Use vcpkg for Dependencies

The official Docker uses vcpkg manifest. Could try:

```dockerfile
RUN git clone https://github.com/microsoft/vcpkg.git
RUN ./vcpkg/bootstrap-vcpkg.sh
RUN cmake .. -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### 5. Use the Official COLMAP Docker Image Directly

Instead of copying binaries, use `colmap/colmap` as a sidecar or switch base image:

```dockerfile
# Option A: Use as base (but lose our CUDA 12.1 base)
FROM colmap/colmap:latest AS base

# Option B: Multi-container setup
# Run COLMAP in separate container, mount shared volume
```

### 6. Match Official Docker's Exact Setup

The official Dockerfile uses:
- Ubuntu 24.04 (not 22.04)
- Different CUDA version
- OpenImageIO from apt

Could try upgrading our base:
```dockerfile
FROM nvidia/cuda:12.4.0-cudnn-devel-ubuntu24.04 AS base
```

**Risk**: May break other dependencies (PyTorch, ComfyUI nodes, etc.)

### 7. Debug Image Reading Directly

Add diagnostic step to verify image format:

```dockerfile
RUN apt-get install -y imagemagick
RUN identify /path/to/test/frame_0001.png
```

And check if COLMAP's image reader works standalone:
```bash
colmap image_reader --image_path /test/frames --database_path /test/db.db
```

### 8. Check OpenImageIO Linking

Verify COLMAP was built with OpenImageIO support:

```bash
ldd /usr/local/bin/colmap | grep -i openimageio
ldd /usr/local/bin/colmap | grep -i freeimage
```

If FreeImage still appears, the build didn't use OpenImageIO.

---

## Diagnostic Commands

```bash
# Inside container:

# Check what image libraries COLMAP is linked against
ldd /usr/local/bin/colmap | grep -iE "(freeimage|openimageio|oiio)"

# Check if images are readable by other tools
file /workspace/projects/*/source/frames/frame_0001.png
identify /workspace/projects/*/source/frames/frame_0001.png

# Check COLMAP version and features
colmap -h
colmap feature_extractor --help | grep -i image

# Test minimal COLMAP image reading
colmap database_creator --database_path /tmp/test.db
colmap feature_extractor --database_path /tmp/test.db --image_path /path/to/frames --SiftExtraction.use_gpu 0
```

---

## References

- [COLMAP Issue #1548](https://github.com/colmap/colmap/issues/1548) - FreeImage initialization bug
- [COLMAP Issue #1845](https://github.com/colmap/colmap/issues/1845) - "Failed to read image file format" with PNG
- [COLMAP PR #1549](https://github.com/colmap/colmap/pull/1549) - Static FreeImage init fix
- [COLMAP PR #2332](https://github.com/colmap/colmap/pull/2332) - FreeImage encapsulation
- [Official COLMAP Dockerfile](https://github.com/colmap/colmap/blob/main/docker/Dockerfile)
- [COLMAP 3.10 Release](https://github.com/colmap/colmap/releases/tag/3.10)
- [COLMAP 3.11 Release](https://github.com/colmap/colmap/releases/tag/3.11.0) - C++17, drops Ubuntu 18.04

---

## Recommended Next Steps

1. **Verify OpenImageIO linking** - Run `ldd` to confirm FreeImage is not linked
2. **Try CMake flags** - Explicitly disable FreeImage: `-DFREEIMAGE_ENABLED=OFF`
3. **Consider Ubuntu 24.04** - Match official Docker's environment
4. **vcpkg approach** - Use COLMAP's preferred dependency management

---

*Last updated: 2026-01-22*
*Branch: claude/fix-colmap-docker-CVk0y*
