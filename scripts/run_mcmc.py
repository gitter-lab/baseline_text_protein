#!/usr/bin/env python3
"""Run 50-step ProTrek-guided MCMC from released Top-1 inputs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.io import atomic_jsonl, iter_jsonl  # noqa: E402
from rethinking_tgpd.mcmc import optimize  # noqa: E402
from rethinking_tgpd.protrek import LocalProTrekScorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protrek-repository", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--final-temperature", type=float, default=0.01)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument(
        "--per-sample-seeds",
        action="store_true",
        help="use global_seed + sample_id; default reproduces the uninterrupted historical global RNG stream",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scorer = LocalProTrekScorer(
        repository=args.protrek_repository,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    historical_rng = random.Random(args.global_seed)
    output = []
    for position, row in enumerate(iter_jsonl(args.input)):
        sample_id = int(row.get("sample_id", position))
        instruction = row.get("instruction")
        top1 = row.get("top1_sequence")
        if not isinstance(instruction, str) or not isinstance(top1, str):
            raise ValueError(f"row {position} needs instruction and top1_sequence")
        text_embedding = scorer.text_embedding(instruction)
        result = optimize(
            top1,
            lambda sequence: scorer.score_with_embedding(sequence, text_embedding),
            steps=args.steps,
            initial_temperature=args.initial_temperature,
            final_temperature=args.final_temperature,
            seed=args.global_seed + sample_id if args.per_sample_seeds else None,
            rng=None if args.per_sample_seeds else historical_rng,
        )
        output.append(
            {
                "sample_id": sample_id,
                "best_sequence": result.pop("best_sequence"),
                **result,
            }
        )
        print(f"[{position + 1}] sample={sample_id} best={output[-1]['best_score']:.4f}")
    atomic_jsonl(args.output, output)
    print(f"wrote {len(output)} MCMC rows to {args.output}")


if __name__ == "__main__":
    main()
