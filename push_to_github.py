#!/usr/bin/env python3
"""Push Heat Transfer MCP Server to GitHub under puran-water organization."""

import os
import sys
import subprocess
import shutil
from pathlib import Path

REPO_URL = "https://github.com/puran-water/heat-transfer-mcp.git"
ORG_NAME = "puran-water"
REPO_NAME = "heat-transfer-mcp"

def run_command(cmd, check=True):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stderr.strip(), e.returncode

def check_prerequisites():
    """Check if git is installed and we're in the right directory."""
    # Check git
    _, code = run_command("git --version", check=False)
    if code != 0:
        print("Error: git is not installed. Please install git first.")
        return False
    
    # Check directory
    if not os.path.exists("server.py") or not os.path.isdir("tools") or not os.path.isdir("utils"):
        print("Error: This script must be run from the heat-transfer-mcp directory")
        return False
    
    return True

def main():
    print("=" * 50)
    print("GitHub Push Script for Heat Transfer MCP")
    print(f"Target: {ORG_NAME} organization")
    print("=" * 50)
    print()
    
    if not check_prerequisites():
        sys.exit(1)
    
    # Optional: Remove archive directory
    if os.path.exists("archive"):
        response = input("Remove archive directory with test files? (y/n): ").lower()
        if response == 'y':
            print("Removing archive directory...")
            shutil.rmtree("archive")
            print("Archive removed.")
    
    # Initialize git if needed
    if not os.path.exists(".git"):
        print("Initializing git repository...")
        run_command("git init")
        print("Git repository initialized.")
    else:
        print("Git repository already initialized.")
    
    # Configure git (optional)
    response = input("\nConfigure git user? (y/n): ").lower()
    if response == 'y':
        name = input("Enter git user name: ")
        email = input("Enter git email: ")
        run_command(f'git config user.name "{name}"')
        run_command(f'git config user.email "{email}"')
        print("Git user configured.")
    
    # Add all files
    print("\nAdding files to git...")
    run_command("git add .")
    output, _ = run_command("git status --short")
    print(output)
    
    # Commit
    print("\nCreating initial commit...")
    commit_message = """Initial commit: Heat Transfer MCP Server with automatic unit conversion

Features:
- 14 specialized heat transfer calculation tools
- Automatic imperial-to-SI unit conversion
- 390+ material properties from VDI/ASHRAE databases
- Temperature-dependent fluid properties
- Weather data integration
- Optimized for wastewater treatment applications"""
    
    run_command(f'git commit -m "{commit_message}"')
    
    # Handle remote
    remotes, _ = run_command("git remote")
    if "origin" in remotes:
        print("\nRemote 'origin' already exists. Removing...")
        run_command("git remote remove origin")
    
    print("\nAdding GitHub remote...")
    run_command(f"git remote add origin {REPO_URL}")
    
    # Ensure we're on main branch
    branch, _ = run_command("git rev-parse --abbrev-ref HEAD")
    if branch == "master":
        print("Renaming master branch to main...")
        run_command("git branch -M main")
    
    # Push to GitHub
    print(f"\nReady to push to: {REPO_URL}")
    print("\nMake sure you have:")
    print("  1. Created the repository on GitHub (or have permissions to create it)")
    print(f"  2. Have push access to the {ORG_NAME} organization")
    print()
    
    response = input("Ready to push? (y/n): ").lower()
    if response == 'y':
        print("\nPushing to GitHub...")
        output, code = run_command("git push -u origin main", check=False)
        
        if code == 0:
            print("\n" + "=" * 50)
            print("Successfully pushed to GitHub!")
            print(f"Repository: {REPO_URL}")
            print("=" * 50)
            print("\nNext steps:")
            print(f"1. Visit {REPO_URL}")
            print("2. Add a description: 'MCP server for thermal engineering calculations with automatic unit conversion'")
            print("3. Add topics: mcp, heat-transfer, thermal-engineering, unit-conversion, wastewater")
            print("4. Enable issues and discussions if needed")
            print("5. Add any additional documentation to the wiki")
        else:
            print(f"\nError pushing to GitHub: {output}")
            print("\nTroubleshooting:")
            print("1. Make sure the repository exists or you have permission to create it")
            print("2. Check your GitHub authentication (SSH keys or HTTPS credentials)")
            print("3. Try creating the repository manually on GitHub first")
    else:
        print("\nPush cancelled. You can push manually later with:")
        print("  git push -u origin main")

if __name__ == "__main__":
    main()