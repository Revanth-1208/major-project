"""Tests for Streamlit application entrypoint."""

import py_compile
from pathlib import Path


def test_app_syntax_compilation():
    """Verify app.py compiles without syntax errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists(), "app.py must exist at project root"
    compiled_path = py_compile.compile(str(app_path), doraise=True)
    assert compiled_path is not None
