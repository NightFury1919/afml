# AFML Project -- Claude Code Memory

## Who this is for
Ethan, a college student building a teaching-oriented Python/Jupyter
implementation of every formula in *Advances in Financial Machine
Learning* (AFML) by Marcos Lopez de Prado, for a student-facing
educational audience. Code quality and clarity matter more than speed
of delivery -- this is a teaching codebase, not a personal exercise.

## Technical environment
- OS: Windows 11. Shell: PowerShell (use `;` to chain commands, NOT
  `&&` -- that's a Bash-ism that fails in PowerShell 5.1).
- Python via Miniconda, `mlfinlab` conda environment (NOT `base` --
  always run `conda activate mlfinlab` before installing packages or
  running tests; `(mlfinlab)` should appear in the prompt prefix).
- Confirmed working versions in `mlfinlab`: Python 3.10.20, pandas
  1.5.3, numpy 1.23.5. Don't assume newer pandas syntax works without
  checking -- `.ffill()` is fine, but don't add new dependencies
  without confirming compatibility.
- Workspace root: `C:\ws\AFML\`. GitHub: `NightFury1919/afml` (a
  separate boss account, `bu-ylee`, cannot access this repo directly --
  any repo-access troubleshooting should check for plain URL/visibility
  mistakes first, not assume IP allowlisting).
- Editor: VS Code with the mlfinlab kernel for Jupyter.

## Project structure & conventions
```
C:\ws\AFML\
├── input_data\              SHARED datasets used by multiple chapters
│                             (e.g. BTCTUSD-trades-2026-03.csv, used by
│                             ch02-ch05+). Do NOT duplicate shared CSVs
│                             into each chapter's own folder.
├── ch02\, ch03\, ch04\, ch05\, ...
│   ├── <topic_package>\     implementation, snake_case .py files,
│   │                        always with __init__.py
│   ├── tests\test_chN.py    pytest, KNOWN expected values (hand-traced
│   │                        or cross-validated), not just shape checks
│   ├── input_data\          ONLY chapter-specific datasets here (e.g.
│   │                        SP98H.txt for ch02 specifically)
│   ├── chapter_N_topic.py    example script, real data
│   ├── chapter_N_topic.ipynb  notebook walkthrough, real data
│   ├── README.md, requirements.txt
```

### Path/portability convention (decided 2026-06-28)
- **`.py` scripts**: derive their own root via `__file__`, e.g.
  `root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))`.
  Never hardcode an absolute path in a script -- this needs to work for
  anyone who clones the repo, any OS, any username.
- **`.ipynb` notebooks**: CAN'T reliably use `__file__`/cwd (Jupyter's
  working directory is not reliably the notebook's own folder -- a
  real, confirmed problem on this machine). Use a single hardcoded
  `AFML_ROOT = r'C:\ws\AFML'` variable at the top of the setup cell,
  with a loud, clearly-labeled comment telling a stranger to edit that
  one line for their machine.
- Older chapters (ch02-ch04) predate this convention and have NOT yet
  been migrated -- don't assume they already follow it unless told so.

### Chat/session convention (decided 2026-06-30)
One chat per day of work -- each conversation maps to one calendar date,
making past-work retrieval reliable (search by date = find that session).
At the END of each day's chat, Ethan says he's done, and Claude generates
a structured end-of-day summary covering what was built, decided, debugged,
and verified during that session. The next chat is a new day -- start fresh,
search the prior conversation to pick up context if needed.

### Notebook output convention (decided 2026-06-30)
Wherever a notebook section would normally just print numbers (e.g.
comparison stats, correlation values, summary metrics), also add an
accompanying chart/plot alongside the text output -- don't replace the
text, add a visual next to it. Applies going forward to all chapters,
not just retroactively to existing ones unless asked.

### TDD workflow (always followed)
1. Read the book's snippet/formula carefully.
2. Implement in Python, snake_case, plain-English comments explaining
   the "why" BEFORE the math/code.
3. Verify against the book's own hand-worked examples where available
   -- exact numeric matches, not "looks reasonable."
4. Write pytest tests with KNOWN expected values. Synthetic data is
   fine for unit tests with known outputs; otherwise prefer real data.
5. Fix any real bugs found this way.
6. Embed full pytest results as comments at the bottom of example .py
   files AND as a markdown cell at the bottom of notebooks -- do this
   proactively once tests pass, don't wait to be asked.

### Real-data-first policy
This codebase is for students, not just personal practice. Notebooks
and example scripts should run against REAL data wherever feasible.
Synthetic data is only acceptable for TDD unit tests with known
expected values. Before declaring a real-data pipeline done, actually
run it against the real CSV and report genuine output -- don't assume
synthetic-shaped test data generalizes.

### Git
Ethan commits/pushes himself after each chapter's full deliverable set
(implementation + tests + notebook + script + README + requirements)
is complete and verified locally.

## Known gotchas (don't relitigate these)
- PowerShell `&&` doesn't work on PowerShell 5.1 -- use `;` or separate
  lines.
- `conda activate mlfinlab` is required before `pip install` or
  `pytest` -- packages installed while in `base` won't be visible.
- Building up a `pd.Series` via repeated item-assignment on an
  initially-EMPTY series (`s[key] = value` in a loop) is fragile across
  pandas versions -- collect into a plain dict first, build the Series
  once at the end instead.
- AFML's own printed code occasionally has real bugs/erratum (e.g. a
  single-line tuple assignment in Ch5 Snippet 5.3 that doesn't evaluate
  the way the book implies under Python's assignment semantics) --
  verify printed snippets against actual language semantics, don't
  assume the book's code is bug-free.

## Working style
- Plain-English explanation -> concrete numerical example with full
  math -> code. Always in that order for new concepts.
- Catches subtle bugs/inconsistencies and will push back -- take these
  questions seriously and investigate rather than reassuring.
- Prefers things verified empirically (actually running code, hand
  tracing values) over claims taken on faith.
- Multiprocessing sweet spot on this machine: 4 threads (6 cores
  available, but reduced fan noise/system load is preferred over the
  marginal extra speed from 6).
