#!/bin/bash
#
# VFX Ingest Platform - Shutdown Script
#
# This script stops the VFX Pipeline web interface gracefully.
#
# Usage:
#   ./stop-platform.sh [OPTIONS]
#
# Options:
#   --remove    Remove container after stopping
#   --help      Show this help message

set -e

CONTAINER_NAME="vfx-ingest-platform"
REMOVE_CONTAINER=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove)
            REMOVE_CONTAINER=true
            shift
            ;;
        --help)
            head -n 12 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "═══════════════════════════════════════════"
echo "  VFX Ingest Platform Shutdown"
echo "═══════════════════════════════════════════"
echo -e "${NC}"

# Check if running locally (PID file exists)
if [ -f ".server.pid" ]; then
    echo "Stopping local server..."
    PID=$(cat .server.pid)

    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✓ Server stopped${NC}"
    else
        echo -e "${YELLOW}⚠ Server was not running${NC}"
    fi

    rm -f .server.pid
fi

# Check if Docker container exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Check if container is running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping Docker container..."
        docker stop "${CONTAINER_NAME}"
        echo -e "${GREEN}✓ Container stopped${NC}"
    else
        echo -e "${YELLOW}⚠ Container was not running${NC}"
    fi

    # Remove container if requested
    if [ "$REMOVE_CONTAINER" = true ]; then
        echo "Removing container..."
        docker rm "${CONTAINER_NAME}"
        echo -e "${GREEN}✓ Container removed${NC}"
    fi
elif docker-compose ps 2>/dev/null | grep -q "vfx"; then
    # If using docker-compose
    echo "Stopping Docker Compose services..."
    docker-compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
else
    echo -e "${YELLOW}⚠ No running containers found${NC}"
fi

echo ""
echo -e "${GREEN}Platform shutdown complete${NC}"
echo ""
echo "To start again, run:"
echo "  ./start-platform.sh"
echo ""
