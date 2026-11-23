"""FastAPI web API and Jinja2 search UI for findex (Lab 7)."""

from findex.web.app import app, create_app, get_index, get_optional_index, get_settings
from findex.web.config import Settings

__all__ = [
    "Settings",
    "app",
    "create_app",
    "get_index",
    "get_optional_index",
    "get_settings",
]
