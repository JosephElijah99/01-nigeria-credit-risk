"""Central configuration loader. Every module reads settings from here."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Resolve a config path relative to the project root and make parent dirs."""
    p = ROOT / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
