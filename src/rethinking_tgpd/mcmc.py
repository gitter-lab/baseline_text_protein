"""Single-substitution Metropolis MCMC for ProTrek-guided stress tests."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def mutate_single(sequence: str, rng: random.Random) -> tuple[str, int, str, str]:
    if not sequence:
        raise ValueError("cannot mutate an empty sequence")
    position = rng.randrange(len(sequence))
    old = sequence[position]
    choices = [residue for residue in AMINO_ACIDS if residue != old]
    new = rng.choice(choices)
    return sequence[:position] + new + sequence[position + 1 :], position, old, new


def optimize(
    initial_sequence: str,
    score: Callable[[str], float],
    *,
    steps: int = 50,
    initial_temperature: float = 1.0,
    final_temperature: float = 0.01,
    seed: int | None = 42,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be positive")
    if initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("temperatures must be positive")

    if rng is None:
        rng = random.Random(seed)
    current_sequence = initial_sequence
    current_score = float(score(current_sequence))
    if not math.isfinite(current_score):
        raise ValueError("initial score is non-finite")
    best_sequence = current_sequence
    best_score = current_score
    accepted = 0
    trajectory = [current_score]

    for step in range(steps):
        temperature = initial_temperature - (
            initial_temperature - final_temperature
        ) * step / steps
        candidate, _, _, _ = mutate_single(current_sequence, rng)
        candidate_score = float(score(candidate))
        if not math.isfinite(candidate_score):
            raise ValueError(f"non-finite candidate score at step {step}")
        delta = candidate_score - current_score
        accept = delta > 0 or rng.random() < math.exp(delta / temperature)
        if accept:
            current_sequence = candidate
            current_score = candidate_score
            accepted += 1
            if current_score > best_score:
                best_sequence = current_sequence
                best_score = current_score
        trajectory.append(best_score)

    return {
        "best_sequence": best_sequence,
        "best_score": best_score,
        "initial_score": trajectory[0],
        "accepted": accepted,
        "rejected": steps - accepted,
        "acceptance_rate": accepted / steps,
        "trajectory": trajectory,
        "steps": steps,
        "seed": seed,
        "initial_temperature": initial_temperature,
        "final_temperature": final_temperature,
    }
