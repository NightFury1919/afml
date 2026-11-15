@echo off
rem find_python.bat -- sets PY to the python.exe of the mlfinlab conda environment. Called by the nightly launchers.
rem Order of lookup:
rem   1. the AFML_PYTHON environment variable, if you set it (full path to python.exe). This always wins.
rem   2. the usual per-user conda locations for an environment named mlfinlab.
rem   3. the all-users conda locations under ProgramData.
rem It never falls back to a bare "python" on purpose: that could be the base environment, which must not run the pipeline.
rem If nothing is found PY is left empty and the caller stops with a message.
set "PY=%AFML_PYTHON%"
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist "%USERPROFILE%\miniconda3\envs\mlfinlab\python.exe" set "PY=%USERPROFILE%\miniconda3\envs\mlfinlab\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\mlfinlab\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\mlfinlab\python.exe"
if not defined PY if exist "%USERPROFILE%\.conda\envs\mlfinlab\python.exe" set "PY=%USERPROFILE%\.conda\envs\mlfinlab\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\envs\mlfinlab\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\envs\mlfinlab\python.exe"
if not defined PY if exist "%ProgramData%\miniconda3\envs\mlfinlab\python.exe" set "PY=%ProgramData%\miniconda3\envs\mlfinlab\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\envs\mlfinlab\python.exe" set "PY=%ProgramData%\anaconda3\envs\mlfinlab\python.exe"
