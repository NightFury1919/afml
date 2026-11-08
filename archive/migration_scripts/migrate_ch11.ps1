# migrate_ch11.ps1
# Ch11 (Dangers of Backtesting) layout migration: ch11/backtest_dangers/ -> ch11/ root
# Run from C:\ws\AFML (repo root), conda env doesn't matter for this script.
#
# Mirrors the ch09/ch10 migration pattern:
#   - script/notebook/README/PNGs move to chapter root
#   - implementation package (pbo.py, test_pbo.py, conftest.py) stays in the subfolder
#   - vestigial __init__.py files removed (see handoff notes: ch11's had a
#     LOAD-BEARING comment claiming otherwise -- verify carefully, see below)

cd C:\ws\AFML

# --- move chapter-root files -------------------------------------------------
git mv ch11\backtest_dangers\chapter_11_backtest_dangers.py ch11\chapter_11_backtest_dangers.py
git mv ch11\backtest_dangers\chapter_11_backtest_dangers.ipynb ch11\chapter_11_backtest_dangers.ipynb
git mv ch11\backtest_dangers\README.md ch11\README.md
git mv ch11\backtest_dangers\ch11_multiple_testing.png ch11\ch11_multiple_testing.png
git mv ch11\backtest_dangers\ch11_pbo.png ch11\ch11_pbo.png
git mv ch11\backtest_dangers\ch11_trial_sharpes.png ch11\ch11_trial_sharpes.png

# --- remove vestigial __init__.py files --------------------------------------
# NOTE: ch11/__init__.py and ch11/backtest_dangers/__init__.py were confirmed
# vestigial in sandbox testing (both isolated and full multi-chapter pytest
# runs, clean pycache, from repo root and from within the subfolder -- all
# passed with both files removed). HOWEVER: test_pbo.py has a pre-existing
# LOAD-BEARING comment claiming a missing ch11/__init__.py once produced
# "No module named 'ch11'" on a clean checkout. This was not reproducible in
# the sandbox, but verify extra carefully on the real machine (see below).
git rm ch11\__init__.py
git rm ch11\backtest_dangers\__init__.py

# --- what stays in ch11\backtest_dangers\ ------------------------------------
#   pbo.py, test_pbo.py, conftest.py   (unchanged, no git mv needed)

git status
