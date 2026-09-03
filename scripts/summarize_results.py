#!/usr/bin/env python3
"""Recompute one benchmark's aggregate table from its per-sample JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("mol_instructions", "cameo")
METRICS = (
    "protrek",
    "plddt",
    "tm_score",
    "residue_lddt",
    "long_range_contact_f1",
    "clash_penalty",
)


def metric_values(rows: list[dict], method: str, metric: str) -> list[float]:
    return [
        float(row[method][metric])
        for row in rows
        if row.get(method, {}).get(metric) is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset_dir = args.results_dir / args.dataset
    output = args.output or dataset_dir / "summary.csv"

    records = []
    path = dataset_dir / "per_sample.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    methods = [key for key in rows[0] if key != "sample_id"]
    for method in methods:
        record = {
            "method": method,
            "samples": len(rows),
            "sequences": sum(
                row[method].get("sequence") is not None for row in rows
            ),
        }
        for metric in METRICS:
            observed = metric_values(rows, method, metric)
            record[f"{metric}_n"] = len(observed)
            record[f"{metric}_mean"] = (
                statistics.fmean(observed) if observed else None
            )
            record[f"{metric}_median"] = (
                statistics.median(observed) if observed else None
            )
            record[f"{metric}_std"] = (
                statistics.pstdev(observed) if observed else None
            )
        plddt = metric_values(rows, method, "plddt")
        record["plddt_gt70_n"] = sum(value > 70 for value in plddt)
        record["plddt_gt90_n"] = sum(value > 90 for value in plddt)
        records.append(record)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
