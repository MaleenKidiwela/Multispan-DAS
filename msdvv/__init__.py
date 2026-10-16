"""Pilot dv/v toolkit for the OOI RCA Nokia multispan DAS data."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    """Config from ``path``, else $MSDVV_CONFIG, else config.yaml (the Jan-2026 snapshot pilot)."""
    path = path or os.environ.get("MSDVV_CONFIG") or ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("file_glob", "*.h5")
    cfg.setdefault("require", None)
    var = os.environ.get("MSDVV_DVV")  # e.g. dvv_24h: overrides of the dvv section (stack length, output folder)
    if var:
        cfg["dvv"] = {**cfg["dvv"], **cfg[var]}
    cfg["dvv"].setdefault("out_subdir", "dvv")
    return cfg


def interspan_cfg(cfg):
    """The 'interspan' config section, overridden by the variant named in $MSDVV_INTERSPAN (e.g. interspan_v2)."""
    xs = dict(cfg["interspan"])
    var = os.environ.get("MSDVV_INTERSPAN")
    if var:
        xs.update(cfg[var])
    return xs
