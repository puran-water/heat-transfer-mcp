"""Require shared helpers from the installed public distribution."""
import importlib

def ensure_engineering_utils_importable():
    importlib.import_module("puran_mcp_support.engineering.pydantic_base")
    importlib.import_module("puran_mcp_support.engineering.safe_paths")
