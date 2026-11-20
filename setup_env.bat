@echo off
echo =======================================================
echo Setting up Smart Crowd AI Environment for VS Code...
echo =======================================================
echo.

if not exist ".venv" (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
)

echo Activating environment and installing dependencies...
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo =======================================================
echo Setup complete!
echo Open this folder in VS Code and press F5, or double-click run.bat
echo =======================================================
pause
