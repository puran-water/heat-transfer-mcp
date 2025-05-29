@echo off
REM GitHub Push Script for Heat Transfer MCP Server
REM Target: https://github.com/puran-water/heat-transfer-mcp

echo ======================================
echo GitHub Push Script for Heat Transfer MCP
echo Target: puran-water organization
echo ======================================
echo.

REM Check if git is installed
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: git is not installed. Please install git first.
    pause
    exit /b 1
)

REM Check if we're in the correct directory
if not exist "server.py" (
    echo Error: This script must be run from the heat-transfer-mcp directory
    pause
    exit /b 1
)

REM Optional: Remove archive directory
set /p remove_archive="Remove archive directory with test files? (y/n): "
if /i "%remove_archive%"=="y" (
    echo Removing archive directory...
    rmdir /s /q archive 2>nul
    echo Archive removed.
)

REM Initialize git repository if not already initialized
if not exist ".git" (
    echo Initializing git repository...
    git init
    echo Git repository initialized.
) else (
    echo Git repository already initialized.
)

REM Configure git (optional)
set /p config_git="Configure git user? (y/n): "
if /i "%config_git%"=="y" (
    set /p git_name="Enter git user name: "
    set /p git_email="Enter git email: "
    git config user.name "%git_name%"
    git config user.email "%git_email%"
    echo Git user configured.
)

REM Add all files
echo.
echo Adding files to git...
git add .
git status --short

REM Commit
echo.
echo Creating initial commit...
git commit -m "Initial commit: Heat Transfer MCP Server with automatic unit conversion" -m "" -m "Features:" -m "- 14 specialized heat transfer calculation tools" -m "- Automatic imperial-to-SI unit conversion" -m "- 390+ material properties from VDI/ASHRAE databases" -m "- Temperature-dependent fluid properties" -m "- Weather data integration" -m "- Optimized for wastewater treatment applications"

REM Check if remote already exists
git remote | findstr "origin" >nul 2>nul
if %errorlevel% equ 0 (
    echo.
    echo Remote 'origin' already exists. Removing...
    git remote remove origin
)

REM Add remote
echo.
echo Adding GitHub remote...
git remote add origin https://github.com/puran-water/heat-transfer-mcp.git

REM Create main branch if on master
for /f %%i in ('git rev-parse --abbrev-ref HEAD') do set current_branch=%%i
if "%current_branch%"=="master" (
    echo Renaming master branch to main...
    git branch -M main
)

REM Push to GitHub
echo.
echo Pushing to GitHub...
echo Target: https://github.com/puran-water/heat-transfer-mcp
echo.
echo This will create a new repository under the puran-water organization.
echo Make sure you have:
echo   1. Created the repository on GitHub (or have permissions to create it)
echo   2. Have push access to the puran-water organization
echo.
set /p ready_push="Ready to push? (y/n): "
if /i "%ready_push%"=="y" (
    git push -u origin main
    echo.
    echo ======================================
    echo Successfully pushed to GitHub!
    echo Repository: https://github.com/puran-water/heat-transfer-mcp
    echo ======================================
    echo.
    echo Next steps:
    echo 1. Visit https://github.com/puran-water/heat-transfer-mcp
    echo 2. Add a description: 'MCP server for thermal engineering calculations with automatic unit conversion'
    echo 3. Add topics: mcp, heat-transfer, thermal-engineering, unit-conversion, wastewater
    echo 4. Update the About section
) else (
    echo Push cancelled. You can push manually later with:
    echo   git push -u origin main
)

pause