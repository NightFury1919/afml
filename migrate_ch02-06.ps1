# ch02-06 layout migration -- git mv script
# Run from C:\ws\AFML (repo root). Preserves git history via git mv/git rm.
# After running this script, copy the edited files from the delivered
# package over the corresponding paths (see MIGRATION_NOTES.md), THEN
# run pytest to verify before committing.

cd C:\ws\AFML

# --- ch02 ---
git mv ch02\tests\test_ch02.py ch02\bars\test_ch02.py
git mv ch02\tests\test_results_ch02.txt ch02\bars\legacy_pytest_run_ch02.txt
Remove-Item ch02\tests -Recurse -Force -ErrorAction SilentlyContinue

# --- ch03 ---
git mv ch03\tests\test_ch03.py ch03\labeling\test_ch03.py
git mv ch03\tests\test_results_ch03.txt ch03\labeling\legacy_pytest_run_ch03.txt
Remove-Item ch03\tests -Recurse -Force -ErrorAction SilentlyContinue

# --- ch04 ---
git mv ch04\tests\test_ch04.py ch04\sample_weights\test_ch04.py
git mv ch04\ch04_timed_v2.py ch04\chapter_4_ntrials_fix.py
Remove-Item ch04\tests -Recurse -Force -ErrorAction SilentlyContinue

# --- ch05 ---
git mv ch05\tests\test_ch05.py ch05\frac_diff\test_ch05.py
git rm ch05\frac_diff\__init__.py
Remove-Item ch05\tests -Recurse -Force -ErrorAction SilentlyContinue

# --- ch06 ---
git mv ch06\tests\test_ch06.py ch06\ensemble\test_ch06.py
git rm ch06\ensemble\__init__.py
Remove-Item ch06\tests -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Moves/deletes done. Now copy the edited files from the delivered package" -ForegroundColor Yellow
Write-Host "over their matching paths (see MIGRATION_NOTES.md), then run pytest." -ForegroundColor Yellow
