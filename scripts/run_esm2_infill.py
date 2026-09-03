#!/usr/bin/env python3
"""Generate four ESM-2 edits and select the highest-ProTrek edit per row."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.esm2 import infill_with_rng, select_oracle_candidate  # noqa: E402
from rethinking_tgpd.io import atomic_jsonl  # noqa: E402
from rethinking_tgpd.protrek import LocalProTrekScorer  # noqa: E402


MASK_RATES = (0.05, 0.10, 0.15, 0.25)
INFILL_ROUNDS = 5
MAXIMUM_ESM_RESIDUES = 1022
GLOBAL_SEED = 42


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or JSONL without reordering rows."""

    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise TypeError("JSON input must be an array of objects")
        return value
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {line_number}")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"input line {line_number} is not an object")
        rows.append(row)
    return rows


def run_row(
    row: dict[str, Any],
    row_index: int,
    *,
    rng: random.Random,
    scorer: Any,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    device: str,
) -> dict[str, Any]:
    """Generate and score the four edits for one row."""

    sample_id = row.get("sample_id", row_index)
    if sample_id != row_index:
        raise ValueError(
            "sample_id must equal row order because the RNG stream is order-dependent"
        )
    instruction = row.get("instruction")
    top1_sequence = row.get("top1_sequence")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"row {row_index}: instruction is missing")
    if not isinstance(top1_sequence, str) or not top1_sequence:
        raise ValueError(f"row {row_index}: top1_sequence is missing")

    work_sequence = top1_sequence[:MAXIMUM_ESM_RESIDUES]
    text_embedding = scorer.text_embedding(instruction)
    edit_records: list[dict[str, Any]] = []
    selector_rows: list[tuple[str, str, float]] = []

    for rate in MASK_RATES:
        edited = infill_with_rng(
            work_sequence,
            rate=rate,
            rounds=INFILL_ROUNDS,
            rng=rng,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        )
        score = float(scorer.score_with_embedding(edited, text_embedding))
        name = f"mask_rate_{rate:.2f}"
        selector_rows.append((name, edited, score))
        edit_records.append(
            {"mask_rate": rate, "sequence": edited, "protrek_score": score}
        )

    selected_name, selected_sequence, selected_score = select_oracle_candidate(
        selector_rows
    )
    return {
        "sample_id": sample_id,
        "edit_candidates": edit_records,
        "selected_mask_rate": float(selected_name.removeprefix("mask_rate_")),
        "selected_sequence": selected_sequence,
        "selected_protrek_score": selected_score,
    }


def run_rows(
    rows: Iterable[dict[str, Any]],
    *,
    scorer: Any,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    device: str,
    global_seed: int = GLOBAL_SEED,
) -> list[dict[str, Any]]:
    rng = random.Random(global_seed)
    return [
        run_row(
            row,
            row_index,
            rng=rng,
            scorer=scorer,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
        )
        for row_index, row in enumerate(rows)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protrek-repository", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument(
        "--esm-revision", default="08e4846e537177426273712802403f7ba8261b6c"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoTokenizer, EsmForMaskedLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install PyTorch and Transformers for ESM-2") from exc

    rows = load_rows(args.input.resolve())
    if not rows:
        raise ValueError("input contains no rows")
    tokenizer = AutoTokenizer.from_pretrained(
        args.esm_model, revision=args.esm_revision
    )
    model = (
        EsmForMaskedLM.from_pretrained(args.esm_model, revision=args.esm_revision)
        .eval()
        .to(args.device)
    )
    scorer = LocalProTrekScorer(
        repository=args.protrek_repository,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    output = run_rows(
        rows,
        scorer=scorer,
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        device=args.device,
        global_seed=args.seed,
    )
    atomic_jsonl(args.output.resolve(), output)
    print(f"wrote {len(output)} ESM-2 rows to {args.output}")


if __name__ == "__main__":
    main()
