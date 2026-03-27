#!/bin/bash

set -eo pipefail

# Verify we're running from project root (check for tiptop-specific files)
if [ ! -f "tiptop/__init__.py" ]; then
    echo "ERROR: This script must be run from the tiptop project root directory"
    echo "Please run: pixi run install-zed"
    exit 1
fi

echo "==> Installing ZED Python API..."

ZED_INSTALLER="/usr/local/zed/get_python_api.py"

if [ ! -f "$ZED_INSTALLER" ]; then
    echo "ERROR: ZED SDK not found at $ZED_INSTALLER"
    echo "Please install the ZED SDK first: https://www.stereolabs.com/docs/development/zed-sdk/linux"
    exit 1
fi

echo "Running ZED Python API installer..."
python "$ZED_INSTALLER"

echo "✓ ZED Python API installed successfully"
