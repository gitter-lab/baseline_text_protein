"""Top-1 PubMedBERT/FAISS retrieval with the historical mapping semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def instruction_to_sequence_last(
    training_rows: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Map duplicate instructions to their final sequence in input order."""
    mapping: dict[str, str] = {}
    for position, row in enumerate(training_rows):
        instruction = row.get("instruction")
        sequence = row.get("sequence")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError(f"training row {position} has no instruction")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(f"training row {position} has no sequence")
        mapping[instruction] = sequence
    return mapping


class Top1Retriever:
    """Thin wrapper around the trusted LangChain FAISS store used in the study."""

    def __init__(
        self,
        *,
        training_rows: list[dict[str, Any]],
        index_path: Path,
        embedding_model: str,
        device: str = "cuda",
        allow_dangerous_deserialization: bool = False,
    ) -> None:
        if not allow_dangerous_deserialization:
            raise ValueError(
                "the FAISS store includes a pickle; verify its hashes, then pass "
                "allow_dangerous_deserialization=True explicitly"
            )
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - optional model stack
            raise RuntimeError(
                "install langchain-community, langchain-huggingface, "
                "sentence-transformers, and a compatible FAISS build"
            ) from exc

        self._mapping = instruction_to_sequence_last(training_rows)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, instruction: str) -> tuple[str, str]:
        documents = self._store.similarity_search(instruction, k=1)
        if not documents:
            raise RuntimeError("FAISS returned no neighbor")
        retrieved_instruction = documents[0].page_content
        try:
            sequence = self._mapping[retrieved_instruction]
        except KeyError as exc:
            raise RuntimeError(
                "retrieved instruction is absent from the training mapping"
            ) from exc
        return retrieved_instruction, sequence

