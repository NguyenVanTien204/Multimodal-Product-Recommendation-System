"""Load `apps/<name>/app` under a unique package name.

`apps/backend/app` and `apps/rag/app` are both packages called `app`, so they cannot
be imported side by side normally. Their internal imports are relative, so loading
each under its own name is enough.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_app_package(alias: str, app_dir: str):
    package_dir = ROOT / "apps" / app_dir / "app"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, package_dir / "__init__.py", submodule_search_locations=[str(package_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module
