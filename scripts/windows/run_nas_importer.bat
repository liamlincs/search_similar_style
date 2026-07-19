@echo off
setlocal
cd /d "%~dp0\..\.."

if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv\Scripts\python.exe
  echo Please create the virtual environment and install project dependencies first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" tools\windows_nas_importer.py --target "Z:\products\standard_samples"
pause
