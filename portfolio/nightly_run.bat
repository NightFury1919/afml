@echo off
set PY=C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
cd /d C:\ws\AFML\portfolio
"%PY%" nightly_run.py >> shadow\nightly_stdout.txt 2>&1
