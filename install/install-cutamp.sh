#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-cutamp"
    exit 1
fi

REPO_URL="https://github.com/tiptop-robot/cuTAMP.git"
INSTALL_DIR="cutamp"

echo "==> Installing cuTAMP..."

# Check if directory exist
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    echo "cutamp already exists at $INSTALL_DIR"

    # Check git integrity
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        echo "✓ cutamp repository is healthy"
        should_clone=false
        cd ..
    else
        echo "✗ cutamp repository is corrupted, removing..."
        cd ..
        rm -rf $INSTALL_DIR
    fi
fi

# Clone
if [ "$should_clone" = true ]; then
    echo "Cloning cuTAMP"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

# Install
cd cutamp
echo "Installing cuTAMP"
pip install -e . --no-build-isolation --no-deps
cd ..
echo "✓ cuTAMP installed successfully"
