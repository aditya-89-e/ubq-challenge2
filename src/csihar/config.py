"""Configuration loading and shared constants."""
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical activity vocabulary shared by both rooms.
ROOM_A_FOLDERS = {
    "fall": "FALL", "sit": "SIT", "sitdown": "SITDOWN",
    "stand": "STAND", "standup": "STANDUP", "walk": "WALK",
}
ROOM_B_PREFIXES = {
    "Empty": "EMPTY", "Lying": "LYING", "Sitting": "SIT",
    "Standing": "STAND", "Walking": "WALK",
}
ROOM_ACTIVITIES = {
    "A": ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"],
    "B": ["SIT", "STAND", "WALK", "EMPTY", "LYING"],
}
SHARED = ["SIT", "STAND", "WALK"]
EXCLUSIVE = {"A": ["SITDOWN", "STANDUP", "FALL"], "B": ["EMPTY", "LYING"]}
ALL_ACTIVITIES = ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL", "EMPTY", "LYING"]
ROOMS = ["A", "B"]


def load_config(path=None):
    path = Path(path) if path else PROJECT_ROOT / "configs" / "experiment.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    return cfg


def resolve(cfg, key):
    """Resolve a path from cfg['paths'] relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def section_hash(cfg, *sections):
    """Short stable hash of config sections, recorded for provenance."""
    blob = json.dumps({s: cfg.get(s) for s in sections}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]
