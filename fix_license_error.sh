#!/bin/bash
echo "=== Fixing license-file error ==="

cd ~/Downloads/fanos_pypi_ready_fixed

# Backup
cp pyproject.toml pyproject.toml.backup

# Check current content
echo "Current license section:"
grep -A2 -B2 "license" pyproject.toml

# Fix 1: Remove license-file if it exists
if grep -q "license-file" pyproject.toml; then
    echo "Removing problematic license-file field..."
    sed -i '/license-file = "LICENSE"/d' pyproject.toml
fi

# Fix 2: Ensure license is specified correctly
if ! grep -q "license" pyproject.toml; then
    echo "Adding proper license field..."
    cat >> pyproject.toml << 'LICENSE'
    
[project]
license = {text = "MIT License"}
LICENSE
fi

# Check fixed version
echo ""
echo "Fixed license section:"
grep -A2 -B2 "license" pyproject.toml

# Rebuild
echo ""
echo "Rebuilding..."
rm -rf dist/ build/ *.egg-info/
python3 -m build

# Upload
echo ""
echo "Uploading..."
python3 -m twine upload dist/*

echo "✅ Fixed and uploaded!"
