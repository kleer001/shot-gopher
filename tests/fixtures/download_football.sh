#!/bin/bash
set -e

# Download Football CIF test video for integration testing

FIXTURES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOOTBALL_Y4M="$FIXTURES_DIR/football_cif.y4m"
FOOTBALL_MP4="$FIXTURES_DIR/football_test.mp4"
FOOTBALL_SHORT="$FIXTURES_DIR/football_short.mp4"

echo "=== Downloading Football CIF Test Video ==="
echo "Fixtures directory: $FIXTURES_DIR"
echo ""

# Download Football CIF if not exists
if [ ! -f "$FOOTBALL_Y4M" ]; then
    echo "Downloading Football CIF (YUV 4:2:0, 352x288, 260 frames)..."
    if command -v wget &> /dev/null; then
        wget -O "$FOOTBALL_Y4M" https://media.xiph.org/video/derf/y4m/football_cif.y4m
    elif command -v curl &> /dev/null; then
        curl -L -o "$FOOTBALL_Y4M" https://media.xiph.org/video/derf/y4m/football_cif.y4m
    else
        echo "ERROR: Neither wget nor curl found"
        exit 1
    fi
    echo "✓ Downloaded Football CIF"
else
    echo "✓ Football CIF already downloaded"
fi

# Convert to MP4 if not exists
if [ ! -f "$FOOTBALL_MP4" ] && [ -f "$FOOTBALL_Y4M" ]; then
    echo "Converting to MP4..."
    if ! command -v ffmpeg &> /dev/null; then
        echo "ERROR: ffmpeg not found. Install ffmpeg to convert test video."
        exit 1
    fi
    ffmpeg -i "$FOOTBALL_Y4M" -c:v libx264 -preset slow -crf 18 "$FOOTBALL_MP4" -y
    echo "✓ Converted to MP4"
else
    echo "✓ MP4 version already exists"
fi

# Create short version (30 frames) if not exists
if [ ! -f "$FOOTBALL_SHORT" ] && [ -f "$FOOTBALL_MP4" ]; then
    echo "Creating short version (30 frames for quick tests)..."
    ffmpeg -i "$FOOTBALL_MP4" -vframes 30 "$FOOTBALL_SHORT" -y
    echo "✓ Created short version"
else
    echo "✓ Short version already exists"
fi

# Trim all videos to 2 seconds
echo "Trimming videos to 2 seconds..."
python3 "$FIXTURES_DIR/trim_videos.py"
echo "✓ Videos trimmed"

echo ""
echo "=== Test Fixtures Ready ==="
echo "Videos trimmed to 2 seconds: $FOOTBALL_MP4, $FOOTBALL_SHORT"
echo ""
