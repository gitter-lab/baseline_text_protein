#!/usr/bin/env python3
"""Remove the fixed instruction template from CAMEO train/test JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rethinking_tgpd.io import atomic_json, load_json  # noqa: E402

PREFIX = "Generate a protein sequence for a novel protein that integrates the following function keywords: "
SUFFIX = ". The designed protein sequence is"


def extract_keywords(instruction: str) -> str:
    text = instruction
    if text.startswith(PREFIX):
        text = text[len(PREFIX):]
    for suffix in (SUFFIX + " ", SUFFIX):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    for split in ("training", "test"):
        rows = load_json(args.data_dir / f"{split}.json")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{split}.json must be a nonempty JSON array")
        prepared = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("instruction"), str):
                raise ValueError(f"{split} row {i} has no instruction")
            if not isinstance(row.get("sequence"), str) or not row["sequence"]:
                raise ValueError(f"{split} row {i} has no sequence")
            instruction = extract_keywords(row["instruction"])
            if not instruction:
                raise ValueError(f"{split} row {i} has no keywords")
            prepared.append({**row, "instruction": instruction})
        path = args.data_dir / f"{split}_keywords.json"
        atomic_json(path, prepared)
        print(f"wrote {len(prepared)} rows to {path}")


if __name__ == "__main__":
    main()
