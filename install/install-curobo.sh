#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-curobo"
    exit 1
fi

REPO_URL="https://github.com/williamshen-nz/curobo.git"
INSTALL_DIR="curobo"

echo "==> Installing curobo..."

# Check if directory exist
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    echo "curobo already exists at $INSTALL_DIR"

    # Check git integrity
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        echo "✓ curobo repository is healthy"
        should_clone=false
        cd ..
    else
        echo "✗ curobo repository is corrupted, removing..."
        cd ..
        rm -rf $INSTALL_DIR
    fi
fi

# Clone and install
if [ "$should_clone" = true ]; then
    echo "Cloning curobo"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Ensure we're on the latest origin/main
cd curobo
echo "Fetching latest from origin/main..."
git fetch origin main
git checkout main
git pull --ff-only origin main
echo "✓ curobo at main (tracking origin/main): $(git rev-parse --short HEAD)"

# pip install (builds CUDA kernels too)
echo "Installing curobo (might take 5-20 minutes)..."
pip install -e . --no-build-isolation --no-deps
cd ..

echo "✓ curobo installed successfully"
