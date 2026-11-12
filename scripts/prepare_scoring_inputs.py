#!/usr/bin/env python3
"""Join generated sequences with test instructions for ProTrek and folding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rethinking_tgpd.io import atomic_jsonl, load_json  # noqa: E402


def load_predictions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("predictions must be a nonempty JSON array or JSONL file")
    return rows


def prepare(test: list[dict], predictions: list[dict], sequence_key: str) -> tuple[list[dict], list[dict]]:
    if not isinstance(test, list) or not test:
        raise ValueError("test must be a nonempty JSON array")
    explicit_ids = ["sample_id" in row for row in predictions]
    if any(explicit_ids) != all(explicit_ids):
        raise ValueError("predictions must either all have sample_id or all use test order")
    if not any(explicit_ids) and len(predictions) != len(test):
        raise ValueError("predictions without sample_id must cover the full test set in order")
    scores, folds, seen = [], [], set()
    for position, row in enumerate(predictions):
        sample_id = row.get("sample_id", position)
        if type(sample_id) is not int or not 0 <= sample_id < len(test) or sample_id in seen:
            raise ValueError(f"invalid or duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        source = test[sample_id]
        instruction, target = source.get("instruction"), source.get("sequence")
        sequence = row.get(sequence_key)
        if not all(isinstance(value, str) and value.strip() for value in (instruction, target, sequence)):
            raise ValueError(f"sample {sample_id}: instruction, reference or {sequence_key} is missing")
        if "instruction" in row and row["instruction"].strip() != instruction.strip():
            raise ValueError(f"sample {sample_id}: prediction instruction does not match the test input")
        if "reference" in row and row["reference"].strip() != target.strip():
            raise ValueError(f"sample {sample_id}: prediction reference does not match the test input")
        scores.append({"sample_id": sample_id, "instruction": instruction, "sequence": sequence})
        folds.append({"sample_id": sample_id, "target": target, "gen": [sequence]})
    return scores, folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="omit to score the reference sequences")
    parser.add_argument("--sequence-key", default="sequence")
    parser.add_argument("--output", type=Path, required=True, help="instruction/sequence JSONL for score_protrek.py")
    parser.add_argument("--folding-output", type=Path, help="optional target/gen JSONL for score_esmfold2.py")
    args = parser.parse_args()
    inputs = {args.test.resolve()}
    if args.predictions:
        inputs.add(args.predictions.resolve())
    outputs = [args.output.resolve()]
    if args.folding_output:
        outputs.append(args.folding_output.resolve())
    if len(outputs) != len(set(outputs)) or any(path in inputs for path in outputs):
        raise ValueError("input and output paths must be distinct")
    test = load_json(args.test)
    predictions = load_predictions(args.predictions) if args.predictions else test
    scores, folds = prepare(test, predictions, args.sequence_key)
    atomic_jsonl(args.output, scores)
    if args.folding_output:
        atomic_jsonl(args.folding_output, folds)
    print(f"prepared {len(scores)} sequences")


if __name__ == "__main__":
    main()
