"""Load configs/default.yaml.

Config is a plain dict passed explicitly into every pass — no module-level
singleton, so a sweep can hold several configs at once without them fighting
over global state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a config file, defaulting to configs/default.yaml."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
