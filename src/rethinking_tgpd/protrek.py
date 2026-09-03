"""Local ProTrek scorer with the paper's explicit 100*cosine convention."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any


class LocalProTrekScorer:
    def __init__(
        self,
        *,
        repository: Path,
        checkpoint_dir: Path,
        device: str = "cuda",
    ) -> None:
        repository = repository.resolve()
        checkpoint_dir = checkpoint_dir.resolve()
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        try:
            import torch
            from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
        except ImportError as exc:  # pragma: no cover - optional model stack
            raise RuntimeError("install the pinned ProTrek environment") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        config = {
            "protein_config": str(checkpoint_dir / "esm2_t33_650M_UR50D"),
            "text_config": str(
                checkpoint_dir
                / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
            ),
            "structure_config": str(checkpoint_dir / "foldseek_t30_150M"),
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "from_checkpoint": str(checkpoint_dir / "ProTrek_650M.pt"),
        }
        self._torch = torch
        self._model = ProTrekTrimodalModel(**config).eval().to(device)

    def text_embedding(self, text: str) -> Any:
        if not text.strip():
            raise ValueError("text is empty")
        with self._torch.inference_mode():
            return self._model.get_text_repr([text], batch_size=1, verbose=False)

    def score_with_embedding(self, sequence: str, text_embedding: Any) -> float:
        if not sequence:
            raise ValueError("sequence is empty")
        with self._torch.inference_mode():
            protein = self._model.get_protein_repr(
                [sequence], batch_size=1, verbose=False
            )
            value = float((protein @ text_embedding.T).item() * 100.0)
        if not math.isfinite(value):
            raise ValueError("ProTrek produced a non-finite score")
        return value

    def score(self, sequence: str, text: str) -> float:
        return self.score_with_embedding(sequence, self.text_embedding(text))

