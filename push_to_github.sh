#!/bin/bash

# GitHub Push Script for Heat Transfer MCP Server
# Target: https://github.com/puran-water/heat-transfer-mcp

echo "======================================"
echo "GitHub Push Script for Heat Transfer MCP"
echo "Target: puran-water organization"
echo "======================================"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "server.py" ] || [ ! -d "tools" ] || [ ! -d "utils" ]; then
    echo "Error: This script must be run from the heat-transfer-mcp directory"
    exit 1
fi

# Optional: Remove archive directory
read -p "Remove archive directory with test files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing archive directory..."
    rm -rf archive/
    echo "Archive removed."
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "Git repository initialized."
else
    echo "Git repository already initialized."
fi

# Configure git (optional)
read -p "Configure git user? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter git user name: " git_name
    read -p "Enter git email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "Git user configured."
fi

# Add all files
echo -e "\nAdding files to git..."
git add .
git status --short

# Commit
echo -e "\nCreating initial commit..."
git commit -m "Initial commit: Heat Transfer MCP Server with automatic unit conversion

Features:
- 14 specialized heat transfer calculation tools
- Automatic imperial-to-SI unit conversion
- 390+ material properties from VDI/ASHRAE databases
- Temperature-dependent fluid properties
- Weather data integration
- Optimized for wastewater treatment applications"

# Check if remote already exists
if git remote | grep -q "origin"; then
    echo -e "\nRemote 'origin' already exists. Removing..."
    git remote remove origin
fi

# Add remote
echo -e "\nAdding GitHub remote..."
git remote add origin https://github.com/puran-water/heat-transfer-mcp.git

# Create main branch if on master
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" = "master" ]; then
    echo "Renaming master branch to main..."
    git branch -M main
fi

# Push to GitHub
echo -e "\nPushing to GitHub..."
echo "Target: https://github.com/puran-water/heat-transfer-mcp"
echo -e "\nThis will create a new repository under the puran-water organization."
echo "Make sure you have:"
echo "  1. Created the repository on GitHub (or have permissions to create it)"
echo "  2. Have push access to the puran-water organization"
echo ""
read -p "Ready to push? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push -u origin main
    echo -e "\n======================================"
    echo "Successfully pushed to GitHub!"
    echo "Repository: https://github.com/puran-water/heat-transfer-mcp"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "1. Visit https://github.com/puran-water/heat-transfer-mcp"
    echo "2. Add a description: 'MCP server for thermal engineering calculations with automatic unit conversion'"
    echo "3. Add topics: mcp, heat-transfer, thermal-engineering, unit-conversion, wastewater"
    echo "4. Update the About section"
else
    echo "Push cancelled. You can push manually later with:"
    echo "  git push -u origin main"
fi