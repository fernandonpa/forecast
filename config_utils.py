from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration into a dictionary."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
