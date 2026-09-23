"""
AutoDataScientist Root Entrypoint for Streamlit Community Cloud and Local Execution.
"""

import sys
from pathlib import Path
import runpy

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if __name__ == "__main__":
    app_file = str(root_dir / "src" / "ui" / "app.py")
    runpy.run_path(app_file, run_name="__main__")
