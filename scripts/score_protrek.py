#!/usr/bin/env python3
"""Score instruction/sequence JSONL pairs with ProTrek."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.io import atomic_jsonl, iter_jsonl  # noqa: E402
from rethinking_tgpd.protrek import LocalProTrekScorer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protrek-repository", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scorer = LocalProTrekScorer(
        repository=args.protrek_repository,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    output = []
    for position, row in enumerate(iter_jsonl(args.input)):
        instruction = row.get("instruction")
        sequence = row.get("sequence")
        if not isinstance(instruction, str) or not isinstance(sequence, str):
            raise ValueError(f"row {position} needs instruction and sequence")
        output.append(
            {
                "sample_id": row.get("sample_id", position),
                "protrek": scorer.score(sequence, instruction),
            }
        )
    atomic_jsonl(args.output, output)
    print(f"wrote {len(output)} scores to {args.output}")


if __name__ == "__main__":
    main()
