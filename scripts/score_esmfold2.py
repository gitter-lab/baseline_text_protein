#!/usr/bin/env python3
"""Score target/generated sequence groups with an explicitly configured API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.esmfold2 import ESMFold2Client, normalize_for_folding  # noqa: E402
from rethinking_tgpd.io import atomic_jsonl, iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="without this flag, validate and print the request count only",
    )
    return parser.parse_args()


def prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = []
    metadata = []
    for position, row in enumerate(rows):
        target = row.get("target")
        generated = row.get("gen")
        if not isinstance(target, str) or not isinstance(generated, list) or not generated:
            raise ValueError(f"row {position} needs target:str and gen:list[str]")
        if not all(isinstance(sequence, str) and sequence for sequence in generated):
            raise ValueError(f"row {position} has an invalid generated sequence")
        normalized_target, target_transform = normalize_for_folding(target)
        normalized_generated = []
        generated_transforms = []
        for sequence in generated:
            normalized, transform = normalize_for_folding(sequence)
            normalized_generated.append(normalized)
            generated_transforms.append(transform)
        requests.append({"target": normalized_target, "gen": normalized_generated})
        metadata.append(
            {
                "sample_id": row.get("sample_id", position),
                "target_transform": target_transform,
                "generated_transform": generated_transforms,
            }
        )
    return requests, metadata


def main() -> None:
    args = parse_args()
    rows = list(iter_jsonl(args.input))
    groups, metadata = prepare(rows)
    request_count = (len(groups) + args.max_groups - 1) // args.max_groups
    if not args.execute:
        print(
            f"validated {len(groups)} groups; would send {request_count} requests "
            f"of at most {args.max_groups} groups"
        )
        return
    client = ESMFold2Client(
        url=os.environ.get("ESMFOLD2_API_URL"),
        token=os.environ.get("ESMFOLD2_API_TOKEN"),
        timeout=args.timeout,
        retries=args.retries,
    )
    outputs = []
    cursor = 0
    for response in client.score_in_batches(groups, max_groups=args.max_groups):
        for group_result in response["groups"]:
            record = {
                "schema_version": 1,
                **metadata[cursor],
                "result": group_result,
            }
            outputs.append(record)
            cursor += 1
    if cursor != len(groups):
        raise RuntimeError("response coverage mismatch")
    atomic_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} scored groups to {args.output}")


if __name__ == "__main__":
    main()
