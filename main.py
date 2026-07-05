"""Top-level Streamlit entrypoint for the thesis demo.
This file launches the dashboard implementation in `tcontext/web_app.py`.
"""
from pathlib import Path
import runpy

APP_ROOT = Path(__file__).resolve().parent
runpy.run_path(str(APP_ROOT / "tcontext" / "web_app.py"), run_name="__main__")
