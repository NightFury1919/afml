# utils/__init__.py
# Shared utilities used across all AFML chapters.
# Import from here so chapter code doesn't need to know the file structure.
#
# Usage:
#   from utils import mp_pandas_obj
#   from utils.multiprocess import mp_pandas_obj

from .multiprocess import mp_pandas_obj, lin_parts
