"""ESM-2 infilling primitives used by the four-rate oracle experiments."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable


VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def rate_key(rate: float) -> str:
    return f"{rate:.2f}"


def infill_with_rng(
    sequence: str,
    *,
    rate: float,
    rounds: int,
    rng: random.Random,
    tokenizer: Any,
    model: Any,
    torch_module: Any,
    device: str,
) -> str:
    """Greedily infill while consuming a caller-owned RNG stream.

    The released Mol-Instructions oracle uses one ``Random(42)`` stream across
    every sample, mask rate, and infill round.  Passing the RNG explicitly
    makes this order-dependent behavior visible and reproducible.  CAMEO uses
    the same stream convention.
    """
    if not sequence:
        raise ValueError("cannot infill an empty sequence")
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must lie in (0, 1]")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    current = list(sequence)
    for _ in range(rounds):
        count = max(1, int(len(current) * rate))
        positions = rng.sample(range(len(current)), min(count, len(current)))
        masked = list(current)
        for position in positions:
            masked[position] = "<mask>"
        inputs = tokenizer(" ".join(masked), return_tensors="pt").to(device)
        with torch_module.no_grad():
            logits = model(**inputs).logits[0]
        for position in positions:
            token_position = position + 1
            if token_position >= logits.shape[0]:
                continue
            token_id = logits[token_position].argmax().item()
            residue = tokenizer.decode([token_id]).strip()
            if len(residue) == 1 and residue in VALID_AA:
                current[position] = residue
    return "".join(current)


def select_oracle_candidate(
    candidates: Iterable[tuple[str, str, float]],
) -> tuple[str, str, float]:
    """Select the first highest-scoring candidate using strict improvement.

    Callers pass only edited candidates, in ascending mask-rate order.  The
    unedited Top-1 sequence is deliberately not a candidate.  A tied edit does
    not replace the incumbent, so exact ties favor the smaller mask rate.
    """
    iterator = iter(candidates)
    try:
        best_name, best_sequence, best_score = next(iterator)
    except StopIteration as exc:
        raise ValueError("oracle selection needs at least one edited candidate") from exc
    if not best_sequence or not math.isfinite(best_score):
        raise ValueError("oracle candidates need a sequence and finite score")
    for name, sequence, score in iterator:
        if not sequence or not math.isfinite(score):
            raise ValueError("oracle candidates need a sequence and finite score")
        if score > best_score:
            best_name, best_sequence, best_score = name, sequence, score
    return best_name, best_sequence, best_score
