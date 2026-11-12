"""Local ESMFold2 inference and residue-level pLDDT scores."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any


CANONICAL_WITH_X = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def normalize_for_folding(sequence: str) -> tuple[str, str]:
    """Uppercase, retain X, map B/U/Z to X, and reject other residues."""
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("folding requires a nonempty sequence")
    normalized: list[str] = []
    transformed = False
    for residue in sequence.upper():
        if residue in CANONICAL_WITH_X:
            normalized.append(residue)
        elif residue in "BUZ":
            normalized.append("X")
            transformed = True
        else:
            raise ValueError(f"unsupported folding residue {residue!r}")
    transforms = ["uppercase"] if sequence != sequence.upper() else []
    if transformed:
        transforms.append("map_buz_to_x")
    return "".join(normalized), "+".join(transforms) or "verbatim"


@dataclass(frozen=True)
class FoldingSettings:
    model: str
    revision: str | None
    num_loops: int
    num_sampling_steps: int
    seed: int
    chunk_size: int
    lm_dropout: float
    esmc_precision: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be a checkpoint directory or model ID")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError("revision must be a string or null")
        for name in ("num_loops", "num_sampling_steps", "chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.lm_dropout, (int, float)) or not 0 <= self.lm_dropout < 1:
            raise ValueError("lm_dropout must be in [0, 1)")
        if self.esmc_precision not in {"fp8", "bf16", "fp32"}:
            raise ValueError("esmc_precision must be fp8, bf16, or fp32")


class LocalESMFold2:
    def __init__(
        self,
        settings: FoldingSettings,
        *,
        device: str = "cuda",
        ccd_cache: Path | None = None,
    ) -> None:
        if sys.version_info < (3, 12):
            raise RuntimeError("Local ESMFold2 requires Python 3.12 or later")
        try:
            import torch
            from esm.models.esmfold2 import (
                ESMFold2InputBuilder,
                EsmFold2Model,
                ProteinInput,
                StructurePredictionInput,
            )
        except ImportError as exc:
            raise RuntimeError("Install folding dependencies with: pip install -e '.[fold]'") from exc
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run local folding on a GPU machine")
        self.settings = settings
        self.device = device
        self._torch = torch
        self._protein_input = ProteinInput
        self._structure_input = StructurePredictionInput
        self.builder = ESMFold2InputBuilder(ccd_cache=ccd_cache)
        self.model = EsmFold2Model.from_pretrained(
            settings.model,
            revision=settings.revision,
            device=device,
            esmc_precision=settings.esmc_precision,
        ).eval()
        self.model.set_chunk_size(settings.chunk_size)
        self.metadata = {
            **asdict(settings),
            "backend": "local",
            "esm_version": version("esm"),
            "device": device,
            "num_diffusion_samples": 1,
        }

    def fold(self, sequence: str, structure_path: Path | None = None) -> dict[str, Any]:
        sequence, _ = normalize_for_folding(sequence)
        started = time.perf_counter()
        inputs = self._structure_input(
            sequences=[self._protein_input(id="A", sequence=sequence)]
        )
        with self._torch.inference_mode():
            prediction = self.builder.fold(
                self.model,
                inputs,
                num_loops=self.settings.num_loops,
                num_sampling_steps=self.settings.num_sampling_steps,
                num_diffusion_samples=1,
                seed=self.settings.seed,
                lm_dropout=self.settings.lm_dropout,
            )
        # MolecularComplex stores per-residue confidence on a 0..1 scale.
        plddt = [100.0 * float(value) for value in prediction.complex.plddt]
        if len(plddt) != len(sequence):
            raise ValueError("predicted residue count does not match the input sequence")
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in plddt):
            raise ValueError("model returned invalid pLDDT values")
        elapsed = time.perf_counter() - started
        if structure_path is not None:
            structure_path.parent.mkdir(parents=True, exist_ok=True)
            structure_path.write_text(prediction.complex.to_mmcif(), encoding="utf-8")
        return {
            "mean_plddt": math.fsum(plddt) / len(plddt),
            "plddt_per_residue": plddt,
            "fold_seconds": elapsed,
            "structure_file": str(structure_path) if structure_path is not None else None,
            "error": None,
        }

    def score_group(
        self, group: dict[str, Any], structure_dir: Path | None = None
    ) -> dict[str, Any]:
        cache: dict[str, dict[str, Any]] = {}

        def score(sequence: str, name: str) -> dict[str, Any]:
            if sequence not in cache:
                path = structure_dir / f"{name}.cif" if structure_dir is not None else None
                try:
                    cache[sequence] = self.fold(sequence, path)
                except (RuntimeError, ValueError) as exc:
                    cache[sequence] = {
                        "mean_plddt": None,
                        "plddt_per_residue": None,
                        "fold_seconds": None,
                        "structure_file": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if self._torch.device(self.device).type == "cuda":
                        self._torch.cuda.empty_cache()
            return cache[sequence]

        target = score(group["target"], "target")
        generated = [score(sequence, f"gen_{i}") for i, sequence in enumerate(group["gen"])]
        return {
            "target_result": target,
            "result": {key: [result[key] for result in generated] for key in target},
        }

