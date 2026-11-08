# migrate_ch12_ch14.ps1
# Ch12 + Ch14 layout migration: chapter root convention (Ch19-onward).
# Run from C:\ws\AFML (repo root).
#
# These two chapters are migrated together because
# chapter_14_backtest_statistics.py/.ipynb import directly from
# chapter_12_cpcv (not cpcv.py) -- that cross-reference has to be updated
# as part of the move, and doing both in one pass lets it be verified in
# one shot rather than half-migrated.

cd C:\ws\AFML

# ============================== Ch12 ========================================
git mv ch12\cpcv\chapter_12_cpcv.py ch12\chapter_12_cpcv.py
git mv ch12\cpcv\chapter_12_cpcv.ipynb ch12\chapter_12_cpcv.ipynb
git mv ch12\cpcv\README.md ch12\README.md
git mv ch12\cpcv\ch12_cpcv_paths.png ch12\ch12_cpcv_paths.png
git mv ch12\cpcv\ch12_cpcv_stats.csv ch12\ch12_cpcv_stats.csv
git mv ch12\cpcv\ch12_cpcv_stats.pkl ch12\ch12_cpcv_stats.pkl

# Vestigial per empirical sandbox testing (isolated + full multi-chapter
# runs, clean caches, both pytest invocation styles). Note: the OLD bare-
# import style genuinely needed ch12/__init__.py (see ch12/cpcv/README.md's
# "Real-machine confirmation" section for the original name-collision bug
# between the cpcv PACKAGE and the cpcv.py MODULE) -- but that bug was in
# the bare-import code, already replaced by the current fully-qualified
# ch12.cpcv.cpcv style, which has no such ambiguity.
git rm ch12\__init__.py
git rm ch12\cpcv\__init__.py

# what stays in ch12\cpcv\: cpcv.py, test_cpcv.py, conftest.py, requirements.txt

# ============================== Ch14 ========================================
git mv ch14\backtest_statistics\chapter_14_backtest_statistics.py ch14\chapter_14_backtest_statistics.py
git mv ch14\backtest_statistics\chapter_14_backtest_statistics.ipynb ch14\chapter_14_backtest_statistics.ipynb
git mv ch14\backtest_statistics\README.md ch14\README.md
git mv ch14\backtest_statistics\ch14_backtest_stats.png ch14\ch14_backtest_stats.png

git rm ch14\__init__.py
git rm ch14\backtest_statistics\__init__.py

# what stays in ch14\backtest_statistics\: backtest_statistics.py,
# classification_scores.py, test_backtest_statistics.py,
# test_classification_scores.py, conftest.py, requirements.txt

git status
