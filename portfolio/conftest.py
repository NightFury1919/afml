"""Lets the tests in portfolio/tests/ import the modules that live in portfolio/ by their plain names.

pytest puts the folder of a conftest.py on sys.path, so `import momentum_ladder` works from any test here.
Run everything with:  python -m pytest portfolio -q
"""
