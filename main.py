"""
Top-level Streamlit entrypoint for Streamlit sharing.
This file delegates execution to the existing app at `tcontext/web_app.py`.
"""
import runpy

runpy.run_path("tcontext/web_app.py", run_name="__main__")
