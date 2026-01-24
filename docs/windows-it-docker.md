# Windows Setup for IT Department (Docker/WSL2)

This document provides IT administrators with the minimal setup requirements to enable users to run the ComfyUI Ingest pipeline on Windows workstations **using Docker and WSL2**. This approach runs the pipeline in a Linux container.

For native Windows installation (no Docker/WSL2), see [windows_for_it_dept_native.md](windows_for_it_dept_native.md).

## Prerequisites

- Windows 10/11 (64-bit)
- NVIDIA GPU with minimum 9 GB VRAM
- Administrator access (for initial setup only)

## Setup Instructions

### 1. Install WSL2 with Ubuntu

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu-22.04
```

**Documentation:** https://learn.microsoft.com/en-us/windows/wsl/install

### 2. Install NVIDIA GPU Driver

Download and install the latest NVIDIA GPU driver for the user's graphics card.

**Download Link:** https://www.nvidia.com/download/index.aspx

### 3. Install Docker Desktop for Windows

Download and install Docker Desktop, ensuring the WSL2 backend is enabled during installation.

**Download Link:** https://docs.docker.com/desktop/install/windows-install/

**Configuration:**
- Enable "Use the WSL 2 based engine" in Docker Desktop settings
- Enable WSL integration for Ubuntu-22.04 distribution

### 4. Configure WSL2 Ubuntu Environment

Launch the Ubuntu-22.04 WSL2 terminal and run the following commands as root/sudo:

```bash
# Add user to docker group (replace <USERNAME> with actual username)
sudo usermod -aG docker <USERNAME>

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg --yes

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit git ffmpeg python3-pip
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Documentation:** https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### 5. Enable Developer Mode (Optional)

This enables symbolic link creation without administrator privileges.

**Steps:**
1. Open Windows Settings
2. Navigate to: Update & Security → For Developers
3. Enable "Developer Mode"

### 6. User Session Restart

Have the user log out and log back in (or restart the computer) to apply the docker group membership.

## Verification

After setup, the user can verify the installation by running in WSL2 Ubuntu:

```bash
# Check Docker works without sudo
docker run hello-world

# Check NVIDIA GPU is accessible
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Both commands should complete successfully without requiring sudo.

## User Instructions

Once IT has completed the above setup, users can bootstrap the pipeline by running in WSL2 Ubuntu:

```bash
curl -fsSL https://raw.githubusercontent.com/kleer001/shot-gopher/main/scripts/bootstrap_docker.sh | bash
```

## Post-Setup User Capabilities

After this one-time setup, users can operate independently without administrator privileges:
- Clone and update the repository
- Download and manage AI models
- Run the Docker-based pipeline
- Install Python packages in their environment
- Process video projects

## Troubleshooting

### Docker Permission Denied
If user gets "permission denied" errors with Docker:
```bash
# Verify user is in docker group
groups | grep docker

# If not listed, IT needs to re-run:
sudo usermod -aG docker <USERNAME>
# Then user must log out and back in
```

### NVIDIA GPU Not Detected
```bash
# Verify NVIDIA driver in Windows
nvidia-smi  # Run in PowerShell

# Verify in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, verify:
1. NVIDIA driver is installed in Windows
2. Docker Desktop has WSL2 integration enabled
3. NVIDIA Container Toolkit is installed in WSL2

## Support

For issues specific to this pipeline, see:
- Troubleshooting guide: [windows-troubleshooting.md](windows-troubleshooting.md)
- Repository: https://github.com/kleer001/shot-gopher
- Issues: https://github.com/kleer001/shot-gopher/issues
