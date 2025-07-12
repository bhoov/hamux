#!/bin/bash

# Modified build script for nbdev project
set -e  # Exit on any error

echo "🔄 Syncing dependencies..."
uv run python scripts/sync_dependencies.py

echo "🤔 nbdev_prepare..."
uv run nbdev_prepare

echo "📄 Copying README from _proc to root directory..."
if [ -f "_proc/README.md" ]; then
    cp _proc/README.md README.md
    echo "✅ README.md successfully updated!"
else
    echo "❌ Warning: _proc/README.md not found. README may not have been generated."
    exit 1
fi

echo "�� Build complete!" 