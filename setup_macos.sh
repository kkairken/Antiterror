#!/bin/bash
# Setup script for macOS Apple Silicon (M1/M2/M3/M4)

set -e

echo "=== AntiTerror Setup for macOS Apple Silicon ==="

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon (arm64)"
    echo "Detected architecture: $(uname -m)"
fi

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS (Metal) support..."
pip install torch torchvision

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements-macos.txt

# Verify MPS availability
echo ""
echo "=== Verifying Installation ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('Apple Silicon GPU acceleration is ready!')
else:
    print('Warning: MPS not available, will use CPU')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the application:"
echo "  source .venv/bin/activate"
echo "  python main.py --device mps"
echo ""
echo "Or with video file:"
echo "  python main.py --source video.mp4 --device mps"
echo ""
echo "For headless servers (no GUI):"
echo "  python main.py --device mps --preview-port 8080"
