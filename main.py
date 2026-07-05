"""Top-level Streamlit entrypoint for the thesis demo.
This file is the single launch target for Streamlit and forwards to the dashboard implementation.
"""
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tcontext.web_app import main as launch_app


def app() -> None:
    """Entry point used by Streamlit when launching main.py."""
    launch_app()


if __name__ == "__main__":
    app()
