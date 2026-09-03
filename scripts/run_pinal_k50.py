#!/usr/bin/env python3
"""Run the audited Pinal K=50 generation and joint-PPL selection protocol."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rethinking_tgpd.io import atomic_jsonl, load_json  # noqa: E402


PROTEIN_PATTERN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{10,}")


def extract_sequence(value: str) -> str:
    matches = PROTEIN_PATTERN.findall(value.upper())
    return max(matches, key=len) if matches else ""


def normalize_prompt(instruction: str, mode: str) -> str:
    value = instruction
    if mode == "mol_instructions_numbered":
        match = re.search(r"1\.", value)
        if match:
            value = value[match.start() :]
    return re.sub(r"\s*The designed protein sequence is\s*$", "", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinal-repository", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("verbatim", "mol_instructions_numbered"),
        default="verbatim",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def load_runtime(repository: Path, weights: Path, seed: int | None) -> dict[str, Any]:
    repository = repository.resolve()
    weights = weights.resolve()
    os.environ["PINAL_MODEL_ROOT"] = str(weights)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    previous = Path.cwd()
    os.chdir(repository)
    try:
        import torch
        import utils.design_utils as design
        from utils.design_utils import (
            SaProtGeneration,
            SaProtPrepareGenerationInputs,
            T2StrcuDefaultGenerationConfig,
            T2StrucPrepareGenerationInputs,
            TO_LIST,
            deduplicate_tuples,
            desc_sanity_check,
            load_pinal,
        )
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        load_pinal()
    finally:
        os.chdir(previous)
    return locals()


def generate(prompt: str, runtime: dict[str, Any]) -> list[tuple[str, float, float]]:
    torch = runtime["torch"]
    design = runtime["design"]
    prompt = runtime["desc_sanity_check"](prompt)
    inputs = runtime["T2StrucPrepareGenerationInputs"](
        design.t2struc,
        prompt,
        design.text_tokenizer,
        design.structre_tokenizer,
    )
    config = runtime["T2StrcuDefaultGenerationConfig"]()
    structures: list[str] = []
    structure_logp: list[float] = []
    for _ in range(10):
        with torch.no_grad():
            generated = design.t2struc.plm.generate(
                do_sample=True,
                generation_config=config,
                num_return_sequences=5,
                return_dict_in_generate=True,
                **inputs,
            )
        structures.extend(
            design.structre_tokenizer.batch_decode(runtime["TO_LIST"](generated.sequences))
        )
        structure_logp.extend(
            torch.sum(generated.log_p, dim=-1).cpu().numpy().tolist()
        )
    saprot_inputs = runtime["SaProtPrepareGenerationInputs"](
        structures,
        prompt,
        design.saprot_text_tokenizer,
        design.saprot_tokenizer,
    )
    decoded = runtime["SaProtGeneration"](
        design.saprot, saprot_inputs, design.saprot_tokenizer
    )
    combined = list(
        zip(structures, structure_logp, decoded["sequence"], decoded["sequence_logp"])
    )
    combined = runtime["deduplicate_tuples"](combined, keyid=2)
    combined.sort(
        key=lambda item: (item[1] + item[3]) / max(len(item[2]), 1), reverse=True
    )
    return [(item[2], float(item[1]), float(item[3])) for item in combined]


def main() -> None:
    args = parse_args()
    rows = load_json(args.input)
    if not isinstance(rows, list):
        raise ValueError("input must be a JSON array")
    if args.limit is not None:
        rows = rows[: args.limit]
    runtime = load_runtime(args.pinal_repository, args.weights, args.seed)
    output = []
    for sample_id, row in enumerate(rows):
        instruction = row.get("instruction")
        if not isinstance(instruction, str):
            raise ValueError(f"invalid input row {sample_id}")
        prompt = normalize_prompt(instruction, args.prompt_mode)
        candidates = generate(prompt, runtime)
        if not candidates:
            raise RuntimeError(f"Pinal emitted no candidate for sample {sample_id}")
        top_five = [
            {
                "sequence": sequence,
                "structure_logp": structure_logp,
                "sequence_logp": sequence_logp,
                "joint_logp_per_residue": (structure_logp + sequence_logp)
                / max(len(sequence), 1),
            }
            for sequence, structure_logp, sequence_logp in candidates[:5]
        ]
        best = extract_sequence(candidates[0][0])
        output.append(
            {
                "sample_id": sample_id,
                "best_sequence": best,
                "saved_candidates": top_five,
            }
        )
        print(f"[{sample_id + 1}/{len(rows)}] length={len(best)}", flush=True)
    atomic_jsonl(args.output, output)
    print(f"wrote {len(output)} Pinal rows to {args.output}")


if __name__ == "__main__":
    main()
