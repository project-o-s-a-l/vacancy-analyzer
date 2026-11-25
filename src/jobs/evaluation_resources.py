"""Read-only review resources in a source checkout or an installed wheel."""

import sys
from pathlib import Path


def evaluation_resource(name):
    if name not in {"benchmark.yaml", "protocol.json", "review.html", "REVIEW.md"}:
        raise ValueError("Unknown evaluation resource")
    root = Path(__file__).resolve().parents[2]
    source = root / ("docs/evaluation" if name == "REVIEW.md" else "config/evaluation") / name
    installed = Path(sys.prefix) / "share/vacancy-analyzer/evaluation" / name
    for path in (source, installed):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing bundled evaluation resource: {name}")
