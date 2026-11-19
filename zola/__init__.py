"""Zola Predict - Local-first football research and prediction assistant."""

import os
import sys

# Ensure Windows 11 DLL search path includes Python DLLs (resolves sqlite3.dll dependency)
if sys.platform == "win32":
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    dlls_dir = os.path.join(base_prefix, "DLLs")
    if os.path.isdir(dlls_dir):
        try:
            os.add_dll_directory(dlls_dir)
        except (AttributeError, OSError):
            pass

__version__ = "0.1.0"
