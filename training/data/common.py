"""Shared helpers for loading training config, used by every script in training/."""
import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    # training/data/../.. -> project root
    return Path(__file__).resolve().parent.parent.parent


def resolve(path_str: str) -> Path:
    """Resolve a config path (relative to project root) into an absolute Path, creating parents."""
    p = project_root() / path_str
    p.mkdir(parents=True, exist_ok=True)
    return p
