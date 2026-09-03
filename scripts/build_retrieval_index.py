#!/usr/bin/env python3
"""Build a normalized PubMedBERT LangChain/FAISS index from JSON rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.io import load_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embedding-model", default="NeuML/pubmedbert-base-embeddings"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--unique-instructions",
        action="store_true",
        help="index each distinct instruction once in first-occurrence order; used for CAMEO keywords",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json(args.training)
    if not isinstance(rows, list):
        raise ValueError("training must be a JSON array")
    texts = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"training row {position} has no instruction")
        if args.unique_instructions and instruction in seen:
            continue
        seen.add(instruction)
        texts.append(instruction)
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise RuntimeError("install the optional retrieval dependencies") from exc
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.from_texts(texts, embeddings)
    args.output.mkdir(parents=True, exist_ok=True)
    store.save_local(str(args.output))
    print(f"indexed {len(texts)} instructions into {args.output}")


if __name__ == "__main__":
    main()

