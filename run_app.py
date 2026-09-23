"""
Convenience launcher for the AutoDataScientist Streamlit dashboard.
Usage:
    python run_app.py
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(__file__), "src", "ui", "app.py")
    print(f"Launching AutoDataScientist Streamlit Dashboard: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd)
