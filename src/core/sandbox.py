"""
Sandboxed code execution engine with subprocess isolation, timeout guards,
and automated Python traceback analysis for self-healing loops.
"""

import os
import sys
import subprocess
import time
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from src.core.config import ExecutionResult, settings

class CodeSandbox:
    def __init__(self, workspace_dir: Optional[str] = None, timeout_seconds: Optional[int] = None):
        self.workspace_dir = Path(workspace_dir or os.getcwd()).resolve()
        self.timeout_seconds = timeout_seconds or settings.sandbox_timeout_seconds
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def execute_script(self, script_path: str, env_vars: Optional[Dict[str, str]] = None) -> ExecutionResult:
        """Executes a python script file in a subprocess and monitors output."""
        script = Path(script_path).resolve()
        if not script.exists():
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"FileNotFoundError: Script {script_path} does not exist.",
                execution_time_seconds=0.0,
                error_summary="File not found",
                error_traceback=f"Target script not found: {script_path}"
            )

        env = os.environ.copy()
        # Add workspace to PYTHONPATH so imports from src work seamlessly
        env["PYTHONPATH"] = str(self.workspace_dir) + os.pathsep + env.get("PYTHONPATH", "")
        if env_vars:
            env.update(env_vars)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return ExecutionResult(
                success=False,
                exit_code=-2,
                stdout=stdout,
                stderr=f"{stderr}\n[ExecutionTimeoutError]: Process exceeded {self.timeout_seconds}s timeout limit.",
                execution_time_seconds=time.time() - start_time,
                error_summary="Execution Timeout",
                error_traceback=f"Process timed out after {self.timeout_seconds} seconds"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                exit_code=-3,
                stdout="",
                stderr=str(e),
                execution_time_seconds=time.time() - start_time,
                error_summary="Subprocess invocation failed",
                error_traceback=str(e)
            )

        exec_time = time.time() - start_time
        success = (exit_code == 0)
        error_summary, error_tb = self._extract_traceback(stderr) if not success else (None, None)
        metrics = self._extract_metrics(stdout, script.parent)
        artifacts = self._discover_artifacts(script.parent)

        return ExecutionResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            execution_time_seconds=exec_time,
            error_summary=error_summary,
            error_traceback=error_tb,
            artifacts_generated=artifacts,
            metrics=metrics
        )

    def execute_code_snippet(self, code_str: str, filename: str = "temp_run.py") -> ExecutionResult:
        """Writes code to a temporary file and executes it."""
        target_path = self.workspace_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code_str)
        return self.execute_script(str(target_path))

    def _extract_traceback(self, stderr: str) -> tuple[str, str]:
        """Extracts the high-level error summary and full traceback from stderr."""
        if not stderr:
            return ("Unknown runtime error", "No stderr produced.")

        # Find typical Python traceback
        tb_match = re.search(r"Traceback \(most recent call last\):.*", stderr, re.DOTALL)
        traceback_str = tb_match.group(0) if tb_match else stderr

        # Extract the last error line, e.g. "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
        lines = [line.strip() for line in stderr.splitlines() if line.strip()]
        last_line = lines[-1] if lines else "Non-zero exit code"
        return (last_line, traceback_str)

    def _extract_metrics(self, stdout: str, search_dir: Path) -> Dict[str, Any]:
        """Looks for metrics printed as JSON or written to metrics.json."""
        metrics_file = search_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Try to find JSON in stdout marker: METRICS_JSON: {...}
        marker = "METRICS_JSON:"
        if marker in stdout:
            try:
                json_part = stdout.split(marker)[-1].splitlines()[0].strip()
                return json.loads(json_part)
            except Exception:
                pass

        return {}

    def _discover_artifacts(self, directory: Path) -> List[str]:
        """Finds newly created model weights, plots, or metrics."""
        patterns = ["*.pt", "*.pth", "*.pkl", "*.joblib", "*.png", "*.json", "*.txt"]
        artifacts = []
        for pat in patterns:
            for f in directory.glob(pat):
                artifacts.append(f.name)
        return list(set(artifacts))
