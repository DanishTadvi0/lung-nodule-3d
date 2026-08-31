"""Tiny config loader: YAML + dotted-key CLI overrides, no external deps beyond PyYAML."""
from __future__ import annotations

import argparse
import ast
import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def load_config(argv: list[str] | None = None) -> dict:
    """Load default.yaml, apply --config FILE, then --key.subkey VALUE overrides.

    Unknown --flags are treated as dotted overrides, so
        --train.lr 1e-3 --model.depth 18
    just works without declaring every field.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    known, extra = parser.parse_known_args(argv)

    with open(known.config, "r") as f:
        cfg = yaml.safe_load(f)

    i = 0
    while i < len(extra):
        token = extra[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if "=" in key:
            key, val = key.split("=", 1)
            _set_dotted(cfg, key, _coerce(val))
            i += 1
        else:
            val = extra[i + 1] if i + 1 < len(extra) else "true"
            _set_dotted(cfg, key, _coerce(val))
            i += 2
    return cfg


def snapshot(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
