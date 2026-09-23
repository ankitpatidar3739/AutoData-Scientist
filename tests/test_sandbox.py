"""
Unit tests for CodeSandbox execution and traceback parsing.
"""

import pytest
import sys
from pathlib import Path
from src.core.sandbox import CodeSandbox

def test_sandbox_successful_execution(tmp_path):
    sandbox = CodeSandbox(workspace_dir=str(tmp_path), timeout_seconds=10)
    code = "print('Hello from sandbox')\n"
    res = sandbox.execute_code_snippet(code, filename="test_ok.py")

    assert res.success is True
    assert res.exit_code == 0
    assert "Hello from sandbox" in res.stdout
    assert res.error_traceback is None

def test_sandbox_captures_traceback_on_error(tmp_path):
    sandbox = CodeSandbox(workspace_dir=str(tmp_path), timeout_seconds=10)
    code = "import sys\nraise ValueError('Custom sandbox error')\n"
    res = sandbox.execute_code_snippet(code, filename="test_fail.py")

    assert res.success is False
    assert res.exit_code != 0
    assert "ValueError" in res.error_summary or "Custom sandbox error" in res.error_summary
    assert "Traceback" in res.error_traceback

def test_sandbox_timeout(tmp_path):
    sandbox = CodeSandbox(workspace_dir=str(tmp_path), timeout_seconds=2)
    code = "import time\ntime.sleep(5)\n"
    res = sandbox.execute_code_snippet(code, filename="test_timeout.py")

    assert res.success is False
    assert res.exit_code == -2
    assert "Timeout" in res.error_summary
