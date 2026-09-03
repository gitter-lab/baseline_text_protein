#!/usr/bin/env python3
"""Run the paper's normalized PubMedBERT/FAISS Top-1 baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.io import atomic_jsonl, load_json  # noqa: E402
from rethinking_tgpd.retrieval import Top1Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--embedding-model",
        default="NeuML/pubmedbert-base-embeddings",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-dangerous-deserialization",
        action="store_true",
        help="required because the LangChain index contains a pickle; use only a trusted index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training = load_json(args.training)
    test = load_json(args.test)
    if not isinstance(training, list) or not isinstance(test, list):
        raise ValueError("training and test inputs must be JSON arrays")
    retriever = Top1Retriever(
        training_rows=training,
        index_path=args.index,
        embedding_model=args.embedding_model,
        device=args.device,
        allow_dangerous_deserialization=args.allow_dangerous_deserialization,
    )
    rows = []
    for sample_id, sample in enumerate(test):
        instruction = sample.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError(f"invalid test row {sample_id}")
        _, sequence = retriever.retrieve(instruction)
        rows.append(
            {
                "sample_id": sample_id,
                "instruction": instruction,
                "top1_sequence": sequence,
            }
        )
    atomic_jsonl(args.output, rows)
    print(f"wrote {len(rows)} Top-1 rows to {args.output}")


if __name__ == "__main__":
    main()
