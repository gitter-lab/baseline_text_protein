"""Authenticated client for the target-aware ESMFold2 scoring contract."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable


CANONICAL_WITH_X = frozenset("ACDEFGHIKLMNPQRSTVWYX")


def normalize_for_folding(sequence: str) -> tuple[str, str]:
    """Retain X; map B/U/Z to X; reject every other residue."""
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
    return "".join(normalized), "map_buz_to_x" if transformed else "verbatim"


class ESMFold2Client:
    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
        timeout: float = 1800.0,
        retries: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self.url = url or os.environ.get("ESMFOLD2_API_URL", "")
        self.token = token or os.environ.get("ESMFOLD2_API_TOKEN", "")
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("set ESMFOLD2_API_URL to an explicit HTTP(S) /score URL")
        if not self.token:
            raise ValueError("ESMFOLD2_API_TOKEN is empty")
        if timeout <= 0 or retries < 1 or backoff_seconds < 0:
            raise ValueError("invalid request retry configuration")
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds

    def score_groups(self, groups: list[dict[str, Any]]) -> dict[str, Any]:
        if not 1 <= len(groups) <= 8:
            raise ValueError("each request must contain 1..8 groups")
        payload = json.dumps({"groups": groups}, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                self.url, data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read())
                if not isinstance(value, dict) or not isinstance(value.get("groups"), list):
                    raise RuntimeError("ESMFold2 response has no groups array")
                if len(value["groups"]) != len(groups):
                    raise RuntimeError("ESMFold2 response group count mismatch")
                return value
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff_seconds * (2**attempt))
        raise RuntimeError("ESMFold2 request failed after retries") from last_error

    def score_in_batches(
        self, groups: Iterable[dict[str, Any]], max_groups: int = 8
    ) -> Iterable[dict[str, Any]]:
        if not 1 <= max_groups <= 8:
            raise ValueError("max_groups must be in 1..8")
        batch: list[dict[str, Any]] = []
        for group in groups:
            batch.append(group)
            if len(batch) == max_groups:
                yield self.score_groups(batch)
                batch = []
        if batch:
            yield self.score_groups(batch)

