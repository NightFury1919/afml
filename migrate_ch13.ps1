# migrate_ch13.ps1
# Ch13 (Optimal Trading Rules) layout migration: ch13/otr/ -> ch13/ root
# Run from C:\ws\AFML (repo root).
# No cross-chapter coupling (unlike Ch12/Ch14) -- straightforward move.

cd C:\ws\AFML

# --- move chapter-root files -------------------------------------------------
git mv ch13\otr\chapter_13_otr.py ch13\chapter_13_otr.py
git mv ch13\otr\chapter_13_otr.ipynb ch13\chapter_13_otr.ipynb
git mv ch13\otr\README.md ch13\README.md
git mv ch13\otr\ch13_book_reproduction.png ch13\ch13_book_reproduction.png
git mv ch13\otr\ch13_real_mesh.png ch13\ch13_real_mesh.png

# --- remove vestigial __init__.py files --------------------------------------
# Verified in sandbox (isolated + full multi-chapter pytest runs, clean
# caches, both from repo root and from within ch13/otr/) that these are
# safe to remove under the current fully-qualified import style, same as
# ch09/ch10/ch12/ch14. test_otr.py's own comment explains WHY the
# __file__-derived pattern is used (robustness against __init__.py-chain
# ambiguity) but does not claim __init__.py is currently required --
# different from ch11's now-stale LOAD-BEARING claim.
git rm ch13\__init__.py
git rm ch13\otr\__init__.py

# --- what stays in ch13\otr\ --------------------------------------------------
#   otr.py, test_otr.py, conftest.py, requirements.txt   (unchanged, no git mv needed)

git status
