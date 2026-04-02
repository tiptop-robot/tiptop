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
REQUIRED_VERSION=$(python -c "
import re
m = re.search(r'REQUIRED_CUTAMP_VERSION = \"([^\"]+)\"', open('tiptop/utils.py').read())
print(m.group(1))
")

echo "==> Installing cuTAMP v$REQUIRED_VERSION..."

# Skip if correct version is already installed
installed_version=$(python -c "import cutamp; print(cutamp.__version__)" 2>/dev/null || echo "")
if [ "$installed_version" = "$REQUIRED_VERSION" ]; then
    echo "✓ cuTAMP v$REQUIRED_VERSION already installed, skipping"
    exit 0
elif [ -n "$installed_version" ]; then
    echo "Installed cuTAMP version ($installed_version) != required ($REQUIRED_VERSION), updating..."
else
    echo "cuTAMP v$REQUIRED_VERSION not installed, installing..."
fi

# Check if directory exists and is a healthy git repo
should_clone=true
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR"
    if git fsck --full &> /dev/null; then
        echo "Updating existing cuTAMP repository to v$REQUIRED_VERSION..."
        git fetch --tags
        git checkout "v$REQUIRED_VERSION"
        should_clone=false
        cd ..
    else
        echo "✗ cutamp repository is corrupted, removing..."
        cd ..
        rm -rf "$INSTALL_DIR"
    fi
fi

# Clone at the required version tag
if [ "$should_clone" = true ]; then
    echo "Cloning cuTAMP v$REQUIRED_VERSION..."
    git clone --branch "v$REQUIRED_VERSION" "$REPO_URL" "$INSTALL_DIR"
fi

# Install
cd "$INSTALL_DIR"
echo "Installing cuTAMP..."
pip install -e . --no-build-isolation --no-deps
cd ..
echo "✓ cuTAMP v$REQUIRED_VERSION installed successfully"
