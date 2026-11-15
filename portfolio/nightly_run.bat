@echo off
rem nightly_run.bat -- price update and shadow scoring only (no accounts). Portable: see find_python.bat.
cd /d "%~dp0"
if not exist shadow mkdir shadow
call "%~dp0find_python.bat"
if not defined PY (
  echo [%date% %time%] FAILED: mlfinlab python not found. Set AFML_PYTHON to its full path. >> shadow\nightly_stdout.txt
  exit /b 1
)
"%PY%" nightly_run.py >> shadow\nightly_stdout.txt 2>&1
