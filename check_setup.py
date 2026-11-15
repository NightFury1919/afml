"""check_setup.py -- one command that tells you whether this machine is ready to run the AFML repo.

    conda activate mlfinlab
    python check_setup.py

Why it exists: the repo is run on more than one computer. Instead of finding out something is wrong
halfway through a nightly run, this prints one PASS / WARN / FAIL line per thing that matters, then runs
a few small real tests. It only reads and checks; it changes nothing and needs no Alpaca keys.

It uses only the standard library, so it still works (and says what is missing) on a half-built machine.
Exit code 0 means no FAIL lines. WARN lines are things worth knowing that do not stop the code.
"""
import importlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))   # the repo folder, wherever it was cloned

# The versions every test result in this repo was produced with (see environment.yml).
PINNED = {"numpy": "1.23.5", "pandas": "1.5.3", "sklearn": "1.2.2", "scipy": "1.15.2", "pytest": "9.0.3"}
# Needed for the tests; missing means FAIL.
REQUIRED = ["numpy", "pandas", "sklearn", "scipy", "pytest", "matplotlib", "requests"]
# Needed only for live data / paper trading; missing means WARN (the chapters still run).
OPTIONAL = ["yfinance", "alpaca"]
# Folders that must be present in a complete clone.
EXPECTED_DIRS = ["ch02", "pipeline", "portfolio", "utils"]


def version_matches(installed, pinned):
    """True if `installed` is the pinned version. A pin of 1.5.3 matches 1.5.3 but not 1.5.30 or 1.5."""
    return installed.strip() == pinned.strip()


def env_name(prefix):
    """The conda environment's folder name, e.g. 'mlfinlab' for C:\\Users\\x\\miniconda3\\envs\\mlfinlab."""
    return prefix.replace("\\", "/").rstrip("/").split("/")[-1]   # handles Windows and POSIX separators on any OS


def check_environment_name(prefix, expected="mlfinlab"):
    """Returns (status, message). The nightly launchers look for the environment called `expected`."""
    name = env_name(prefix)
    if name == "base" or name.lower() in ("miniconda3", "anaconda3", "miniconda", "anaconda"):
        return "FAIL", f"running from the conda base environment ({prefix}); the project must never run there"
    if name != expected:
        return "WARN", f"environment is called '{name}', not '{expected}'; set AFML_PYTHON for the nightly launchers"
    return "PASS", f"environment '{name}'"


def check_python(version_info):
    """The repo's results were produced on Python 3.10."""
    got = f"{version_info[0]}.{version_info[1]}"
    if (version_info[0], version_info[1]) == (3, 10):
        return "PASS", f"Python {got}"
    return "WARN", f"Python {got}; every result in the repo was produced on 3.10"


def _module_version(name):
    try:
        mod = importlib.import_module(name)
    except Exception as e:                      # not installed, or broken install
        return None, f"{type(e).__name__}: {e}"
    return getattr(mod, "__version__", "unknown"), None


def check_package(name, required):
    ver, err = _module_version(name)
    if ver is None:
        return ("FAIL" if required else "WARN"), f"{name} not importable ({err})"
    pin = PINNED.get(name)
    if pin and not version_matches(ver, pin):
        return "WARN", f"{name} {ver} (the repo was tested with {pin})"
    return "PASS", f"{name} {ver}"


def run_small_tests():
    """Run a few fast, real tests as a smoke test. Returns (status, message)."""
    targets = [os.path.join("portfolio", "tests", "test_spy_since.py"),
               os.path.join("tests", "test_check_setup.py")]
    targets = [t for t in targets if os.path.exists(os.path.join(ROOT, t))]
    if not targets:
        return "WARN", "no smoke tests found to run"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + targets
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    last = (res.stdout.strip().splitlines() or ["(no output)"])[-1]
    return ("PASS" if res.returncode == 0 else "FAIL"), f"smoke tests: {last}"


def main():
    results = []
    results.append(("python", *check_python(sys.version_info)))
    results.append(("environment", *check_environment_name(sys.prefix)))
    for name in REQUIRED:
        results.append((name, *check_package(name, True)))
    for name in OPTIONAL:
        results.append((name, *check_package(name, False)))
    for d in EXPECTED_DIRS:
        ok = os.path.isdir(os.path.join(ROOT, d))
        results.append((f"folder {d}", "PASS" if ok else "FAIL", "present" if ok else "missing: incomplete clone?"))
    if not any(s == "FAIL" for _, s, _ in results):
        results.append(("tests", *run_small_tests()))
    width = max(len(r[0]) for r in results)
    for label, status, msg in results:
        print(f"[{status:4}] {label:<{width}}  {msg}")
    fails = sum(1 for _, s, _ in results if s == "FAIL")
    warns = sum(1 for _, s, _ in results if s == "WARN")
    print(f"\n{fails} fail, {warns} warn. " + ("Ready." if fails == 0 else "Fix the FAIL lines first."))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# TDD results (python -m pytest tests/test_check_setup.py -v), run 2026-11-15 in the Claude sandbox
# (Python 3.13, newer pandas than mlfinlab; these tests touch no pandas). Re-run on your machine in the mlfinlab env.
# Mutation check: removing the backslash handling in env_name() made test_env_name_windows_style_path fail.
# ---------------------------------------------------------------------------
# ============================= test session starts ==============================
# platform linux -- Python 3.13.18, pytest-9.1.1, pluggy-1.6.0 -- /usr/local/bin/python3
# rootdir: /home/claude/afml
# plugins: platformdirs-4.13.5, anyio-4.15.1
# collecting ... collected 11 items
# 
# tests/test_check_setup.py::test_exact_version_matches PASSED             [  9%]
# tests/test_check_setup.py::test_version_prefix_is_not_a_match PASSED     [ 18%]
# tests/test_check_setup.py::test_env_name_windows_style_path PASSED       [ 27%]
# tests/test_check_setup.py::test_env_name_posix_path_and_trailing_slash PASSED [ 36%]
# tests/test_check_setup.py::test_base_environment_is_a_fail PASSED        [ 45%]
# tests/test_check_setup.py::test_correct_environment_name_passes PASSED   [ 54%]
# tests/test_check_setup.py::test_other_environment_name_warns PASSED      [ 63%]
# tests/test_check_setup.py::test_python_310_passes_and_other_warns PASSED [ 72%]
# tests/test_check_setup.py::test_missing_required_package_fails_and_optional_warns PASSED [ 81%]
# tests/test_check_setup.py::test_wrong_pinned_version_warns PASSED        [ 90%]
# tests/test_check_setup.py::test_matching_pinned_version_passes PASSED    [100%]
# 
# ============================== 11 passed in 0.02s ==============================
