"""Make the src/ layout importable during tests without requiring an install.

`pip install -e .` is the recommended setup, but this conftest lets `pytest`
discover the `src/` packages even in a fresh clone that hasn't been installed.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
