#!/bin/bash
#
# VFX Ingest Platform - One-Click Startup Script
#
# This script starts the VFX Pipeline web interface and opens it in your browser.
# It handles Docker startup, health checks, and cross-platform browser opening.
#
# Usage:
#   ./start-platform.sh [OPTIONS]
#
# Options:
#   --port PORT      Use custom port (default: 5000)
#   --no-browser     Don't open browser automatically
#   --dev            Start in development mode with hot reload
#   --help           Show this help message

set -e

# Default configuration
PORT="${VFX_PORT:-5000}"
CONTAINER_NAME="vfx-ingest-platform"
OPEN_BROWSER=true
DEV_MODE=false

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --help)
            head -n 15 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "═══════════════════════════════════════════"
echo "  VFX Ingest Platform Startup"
echo "═══════════════════════════════════════════"
echo -e "${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check if container already exists and is running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}⚠ Platform is already running${NC}"
        URL="http://localhost:${PORT}"
    else
        echo "Starting existing container..."
        docker start "${CONTAINER_NAME}"
        URL="http://localhost:${PORT}"
    fi
else
    # Check if docker-compose.yml exists
    if [ -f "docker-compose.yml" ]; then
        echo "Starting platform with Docker Compose..."
        if [ "$DEV_MODE" = true ]; then
            docker-compose up -d --build
        else
            docker-compose up -d
        fi
        URL="http://localhost:${PORT}"
    # Check if Dockerfile exists
    elif [ -f "Dockerfile" ]; then
        echo "Building and starting Docker container..."
        docker build -t vfx-ingest-platform .
        docker run -d \
            --name "${CONTAINER_NAME}" \
            -p "${PORT}:5000" \
            -v "$(pwd)/projects:/projects" \
            -v "$(pwd)/models:/models" \
            vfx-ingest-platform
        URL="http://localhost:${PORT}"
    else
        # Run locally without Docker
        echo -e "${YELLOW}⚠ No Docker configuration found${NC}"
        echo "Starting platform locally..."

        # Check if Python is available
        if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
            echo -e "${RED}Error: Python is not installed${NC}"
            exit 1
        fi

        # Determine Python command
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            PYTHON_CMD="python"
        fi

        # Install requirements if needed
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            $PYTHON_CMD -m venv venv
        fi

        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        fi

        # Install/update requirements
        echo "Installing dependencies..."
        pip install -q -r requirements.txt

        # Start server in background
        echo "Starting web server..."
        cd web
        uvicorn server:app --host 0.0.0.0 --port "${PORT}" &
        SERVER_PID=$!
        cd ..

        # Store PID for shutdown
        echo $SERVER_PID > .server.pid

        URL="http://localhost:${PORT}"
    fi
fi

# Wait for server to be ready
echo "Waiting for server to start..."
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s "${URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is ready${NC}"
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    echo -n "."
    sleep 1
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo -e "\n${RED}Error: Server failed to start within 30 seconds${NC}"
    echo "Check the logs for more information:"
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "  docker logs ${CONTAINER_NAME}"
    else
        echo "  Check console output above"
    fi
    exit 1
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Platform is ready!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Web Interface:${NC} ${URL}"
echo -e "  ${BLUE}API Docs:${NC}      ${URL}/api/docs"
echo ""

# Open browser
if [ "$OPEN_BROWSER" = true ]; then
    echo "Opening browser..."

    # Detect OS and open browser
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$URL"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v xdg-open &> /dev/null; then
            xdg-open "$URL" &> /dev/null
        elif command -v gnome-open &> /dev/null; then
            gnome-open "$URL" &> /dev/null
        else
            echo -e "${YELLOW}⚠ Could not open browser automatically${NC}"
            echo "Please open: $URL"
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash, Cygwin, WSL)
        if command -v cmd.exe &> /dev/null; then
            cmd.exe /c start "$URL"
        elif command -v powershell.exe &> /dev/null; then
            powershell.exe -c "Start-Process '$URL'"
        else
            echo -e "${YELLOW}⚠ Could not open browser automatically${NC}"
            echo "Please open: $URL"
        fi
    else
        echo -e "${YELLOW}⚠ Unknown OS - could not open browser automatically${NC}"
        echo "Please open: $URL"
    fi
fi

echo ""
echo -e "${YELLOW}Press Ctrl+C to view shutdown instructions${NC}"
echo ""

# Trap Ctrl+C to show shutdown instructions
trap 'echo ""; echo "To stop the platform:"; echo "  ./stop-platform.sh"; echo "  OR: Use the Shutdown button in the web interface"; exit 0' INT

# If running locally, wait for server process
if [ -f ".server.pid" ]; then
    wait $(cat .server.pid)
else
    # If running in Docker, show logs
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Showing container logs (Ctrl+C to detach):"
        echo "─────────────────────────────────────────────"
        docker logs -f "${CONTAINER_NAME}" 2>&1
    fi
fi
