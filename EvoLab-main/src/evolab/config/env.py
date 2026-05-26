from __future__ import annotations

import re
from pathlib import Path


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def lookup_env_value(values: dict[str, str], key: str) -> str | None:
    if key in values:
        return values[key]
    lower_key = key.lower()
    for env_key, value in values.items():
        if env_key.lower() == lower_key:
            return value
    return None


def env_ref_prefix(env_ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", env_ref).strip("_").upper()
