from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_size_bytes(path: str | Path) -> int:
    source = Path(path)
    if source.is_file():
        return source.stat().st_size
    if source.is_dir():
        return sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
    return 0


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def environment_info() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform()}
