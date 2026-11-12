#!/usr/bin/env python3
"""Fold target/generated sequences locally and save pLDDT scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.esmfold2 import (  # noqa: E402
    FoldingSettings, LocalESMFold2, normalize_for_folding,
)
from rethinking_tgpd.io import atomic_jsonl, iter_jsonl, load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/esmfold2.json")
    parser.add_argument("--model", help="override the model ID or use a local checkpoint directory")
    parser.add_argument("--revision", help="model revision; overrides the config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-loops", type=int)
    parser.add_argument("--num-sampling-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--structures-dir", type=Path, help="also save predicted structures as mmCIF")
    parser.add_argument("--ccd-cache", type=Path, help="optional directory containing ccd.pkl")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="without this flag, validate inputs and settings without loading the model",
    )
    return parser.parse_args()


def prepare(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = []
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
        groups.append({"target": normalized_target, "gen": normalized_generated})
        metadata.append(
            {
                "sample_id": row.get("sample_id", position),
                "target_transform": target_transform,
                "generated_transform": generated_transforms,
            }
        )
    return groups, metadata


def load_settings(args: argparse.Namespace) -> FoldingSettings:
    settings = load_json(args.config)
    if not isinstance(settings, dict):
        raise ValueError("folding config must be a JSON object")
    if args.model is not None and args.model != settings.get("model"):
        settings["revision"] = None
    for name in ("model", "revision", "num_loops", "num_sampling_steps", "seed", "chunk_size"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    return FoldingSettings(**settings)


def main() -> None:
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    settings = load_settings(args)
    rows = list(iter_jsonl(args.input))
    groups, metadata = prepare(rows)
    if not groups:
        raise ValueError("input has no sequence groups")
    fold_count = sum(len(set([group["target"], *group["gen"]])) for group in groups)
    if not args.execute:
        print(
            f"validated {len(groups)} groups; {fold_count} local folds; "
            f"model={settings.model}, loops={settings.num_loops}, "
            f"sampling_steps={settings.num_sampling_steps}, seed={settings.seed}"
        )
        return
    scorer = LocalESMFold2(settings, device=args.device, ccd_cache=args.ccd_cache)
    outputs = []
    failed = 0
    for position, (group, row_metadata) in enumerate(zip(groups, metadata)):
        structure_dir = args.structures_dir / str(position) if args.structures_dir else None
        scores = scorer.score_group(group, structure_dir)
        outputs.append({
            "schema_version": 2,
            **row_metadata,
            "folding": scorer.metadata,
            **scores,
        })
        errors = [scores["target_result"]["error"], *scores["result"]["error"]]
        failed += sum(error is not None for error in errors)
        for error in set(errors) - {None}:
            print(f"group {position}: {error}", file=sys.stderr)
        print(f"folded group {position + 1}/{len(groups)}", flush=True)
    atomic_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} scored groups to {args.output}; {failed} failed sequence entries")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
