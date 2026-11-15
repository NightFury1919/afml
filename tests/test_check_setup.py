"""Tests for check_setup.py. Known inputs, known answers: no network, no real environment needed."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import check_setup as cs


def test_exact_version_matches():
    assert cs.version_matches("1.5.3", "1.5.3")


def test_version_prefix_is_not_a_match():
    # 1.5.30 must not pass for a 1.5.3 pin, and 1.5 must not either.
    assert not cs.version_matches("1.5.30", "1.5.3")
    assert not cs.version_matches("1.5", "1.5.3")


def test_env_name_windows_style_path():
    assert cs.env_name("C:\\Users\\x\\miniconda3\\envs\\mlfinlab") == "mlfinlab"
    assert cs.check_environment_name("C:\\Users\\x\\miniconda3")[0] == "FAIL"   # a Windows base env is caught too


def test_env_name_posix_path_and_trailing_slash():
    assert cs.env_name("/opt/miniconda3/envs/mlfinlab/") == "mlfinlab"


def test_base_environment_is_a_fail():
    status, _ = cs.check_environment_name("/opt/miniconda3")
    assert status == "FAIL"


def test_correct_environment_name_passes():
    assert cs.check_environment_name("/opt/miniconda3/envs/mlfinlab")[0] == "PASS"


def test_other_environment_name_warns():
    assert cs.check_environment_name("/opt/miniconda3/envs/other")[0] == "WARN"


def test_python_310_passes_and_other_warns():
    assert cs.check_python((3, 10, 20))[0] == "PASS"
    assert cs.check_python((3, 13, 1))[0] == "WARN"


def test_missing_required_package_fails_and_optional_warns():
    assert cs.check_package("definitely_not_a_real_package_xyz", True)[0] == "FAIL"
    assert cs.check_package("definitely_not_a_real_package_xyz", False)[0] == "WARN"


def test_wrong_pinned_version_warns(monkeypatch):
    monkeypatch.setattr(cs, "_module_version", lambda name: ("9.9.9", None))
    assert cs.check_package("pandas", True)[0] == "WARN"


def test_matching_pinned_version_passes(monkeypatch):
    monkeypatch.setattr(cs, "_module_version", lambda name: ("1.5.3", None))
    assert cs.check_package("pandas", True)[0] == "PASS"
