#!/usr/bin/env python3
"""Rebuild the Mol-Instructions subset, difficulty, and UniRef50 tables."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ("training_overlap", "retrieval_quartiles", "uniref50")
METHODS = {
    "reference": "reference",
    "mcmc": "mcmc",
    "esm2_infill": "esm2_oracle",
    "top1": "top1",
    "pinal_k50": "pinal_k50",
    "prodva": "prodva",
}
QUARTILE_METHODS = ("reference", "top1", "pinal_k50", "prodva")
NOVELTY_METHODS = (
    "reference", "top1", "esm2_infill", "mcmc", "pinal_k10", "pinal_k50", "prodva"
)
HIT_FIELDS = ("target", "pident", "evalue", "bits", "qcov", "tcov", "alnlen")


def number(value: str | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite measurement: {value!r}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"invalid CSV header: {path}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"empty or malformed CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", delete=False) as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.chmod(0o644)
    os.replace(temporary, path)


def load_main(path: Path) -> dict[int, dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records or [row["sample_id"] for row in records] != list(range(len(records))):
        raise ValueError("main results must have unique, ordered sample IDs starting at zero")
    return {row["sample_id"]: row for row in records}


def check_scores(rows: list[dict], main: dict[int, dict], methods: tuple | dict) -> list[int]:
    ids = [int(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= main.keys():
        raise ValueError("duplicate or unknown sample ID")
    for row, sample_id in zip(rows, ids):
        for method in methods:
            if number(row[method]) != number(main[sample_id][METHODS[method]]["protrek"]):
                raise ValueError(f"sample {sample_id}: stale {method} ProTrek score")
    return ids


def percent(count: float, total: float) -> float | None:
    return 100.0 * count / total if total else None


def training_summary(rows: list[dict], main: dict[int, dict]) -> list[dict]:
    check_scores(rows, main, METHODS)
    output = []
    for method, key in METHODS.items():
        values = [number(row[method]) for row in rows]
        full = [number(row[key]["protrek"]) for row in main.values()]
        mean, median = statistics.fmean(values), statistics.median(values)
        full_mean, full_median = statistics.fmean(full), statistics.median(full)
        output.append({
            "method": method,
            "n": len(values),
            "protrek_mean": mean,
            "protrek_median": median,
            "protrek_sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "full_n": len(full),
            "full_mean": full_mean,
            "full_median": full_median,
            "mean_delta": mean - full_mean,
            "mean_delta_pct": percent(mean - full_mean, full_mean),
            "median_delta": median - full_median,
            "median_delta_pct": percent(median - full_median, full_median),
        })
    return output


def quartile_summary(rows: list[dict], main: dict[int, dict]) -> list[dict]:
    ids = check_scores(rows, main, QUARTILE_METHODS)
    if set(ids) != main.keys() or len(rows) < 2:
        raise ValueError("difficulty analysis must cover the full benchmark")
    similarities = [number(row["text_similarity"]) for row in rows]
    cutoffs = statistics.quantiles(similarities, n=4, method="inclusive")
    groups: dict[str, list[dict]] = {f"Q{i}": [] for i in range(1, 5)}
    for row, value in zip(rows, similarities):
        quartile = f"Q{bisect_left(cutoffs, value) + 1}"
        if row["quartile"] != quartile:
            raise ValueError(f"sample {row['sample_id']}: incorrect quartile")
        groups[quartile].append(row)
    output = []
    for quartile, members in groups.items():
        observed = [number(row["text_similarity"]) for row in members]
        output.append({
            "quartile": quartile,
            "n": len(members),
            "text_similarity_min": min(observed) if observed else None,
            "text_similarity_max": max(observed) if observed else None,
            **{
                f"{method}_mean": statistics.fmean(number(row[method]) for row in members)
                if members else None
                for method in QUARTILE_METHODS
            },
        })
    return output


def validate_novelty(rows: list[dict], main: dict[int, dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {method: [] for method in NOVELTY_METHODS}
    seen = set()
    for row in rows:
        method, sample_id = row["method"], int(row["sample_id"])
        if method not in groups or sample_id not in main or (method, sample_id) in seen:
            raise ValueError("unknown method/sample or duplicate novelty record")
        seen.add((method, sample_id))
        status = row["search_status"]
        if status == "not_queried":
            if any(row[key] for key in ("query_id", "sequence_matches_current", "query_length", *HIT_FIELDS)):
                raise ValueError("unqueried record contains a query or a hit")
        elif status in ("hit", "no_hit"):
            if not row["query_id"] or int(row["query_length"]) < 1:
                raise ValueError("queried record has no query or sequence length")
            if row["sequence_matches_current"] not in ("true", "false"):
                raise ValueError("invalid sequence-match flag")
            if status == "hit":
                if not row["target"] or not 0 <= number(row["pident"]) <= 100:
                    raise ValueError("invalid identity or target")
                if number(row["evalue"]) < 0 or number(row["bits"]) < 0 or int(row["alnlen"]) < 1:
                    raise ValueError("invalid alignment statistics")
                if any(not 0 <= number(row[key]) <= 1 for key in ("qcov", "tcov")):
                    raise ValueError("invalid alignment coverage")
            elif any(row[key] for key in HIT_FIELDS):
                raise ValueError("no-hit record contains alignment values")
        else:
            raise ValueError(f"unknown search status: {status}")
        groups[method].append(row)
    if any({int(row["sample_id"]) for row in group} != main.keys() for group in groups.values()):
        raise ValueError("each method must include every benchmark ID, including unqueried IDs")
    return groups


def novelty_summary(groups: dict[str, list[dict]], *, current_only: bool) -> list[dict]:
    output = []
    for method, rows in groups.items():
        queried = [row for row in rows if row["search_status"] != "not_queried"]
        different = sum(row["sequence_matches_current"] == "false" for row in queried)
        selected = [row for row in queried if row["sequence_matches_current"] == "true"] if current_only else queried
        hits = [number(row["pident"]) for row in selected if row["search_status"] == "hit"]
        no_hit = sum(row["search_status"] == "no_hit" for row in selected)
        gt90, lt30 = sum(value > 90 for value in hits), sum(value < 30 for value in hits)
        output.append({
            "method": method,
            "benchmark_n": len(rows),
            "queried_n": len(selected),
            "with_hit_n": len(hits),
            "no_hit_n": no_hit,
            "unavailable_n": len(rows) - len(selected),
            "not_queried_n": len(rows) - len(queried),
            "different_sequence_n": different,
            "mean_identity": statistics.fmean(hits) if hits else None,
            "median_identity": statistics.median(hits) if hits else None,
            "identity_population_std": statistics.pstdev(hits) if hits else None,
            "gt90_n": gt90,
            "gt90_pct": percent(gt90, len(selected)),
            "lt30_n": lt30,
            "lt30_pct": percent(lt30, len(selected)),
            "no_hit_pct": percent(no_hit, len(selected)),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", nargs="?", choices=(*ANALYSES, "all"), default="all")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/mol_instructions")
    parser.add_argument("--output-dir", type=Path, help="write summaries under another directory")
    args = parser.parse_args()
    main_rows = load_main(args.results_dir / "per_sample.jsonl")
    destination = args.output_dir or args.results_dir
    for analysis in ANALYSES if args.analysis == "all" else (args.analysis,):
        rows = read_csv(args.results_dir / analysis / "per_sample.csv")
        if analysis == "training_overlap":
            summary = training_summary(rows, main_rows)
        elif analysis == "retrieval_quartiles":
            summary = quartile_summary(rows, main_rows)
        else:
            groups = validate_novelty(rows, main_rows)
            summary = novelty_summary(groups, current_only=True)
            write_csv(destination / analysis / "cached_summary.csv", novelty_summary(groups, current_only=False))
        write_csv(destination / analysis / "summary.csv", summary)
        print(f"wrote {analysis} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
