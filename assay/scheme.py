"""Marker interface (SPEC.md § Data Model) — a Scheme is a (mark, probe_set) pair
keyed by a secret. mark injects a keyed trace into a corpus; probe_set yields the
detector's queries for a key. v0.1 ships the trap-street reference scheme.

trap-street: keyed fabricated facts (entity->year bindings) from real common words,
so per-token perplexity stays low and no single word is novel — the fabrication
lives only in the combination (the plausible-payload constraint, SPEC Decision 3).
A suspect that completes the fabrication was trained on the marked corpus."""

from __future__ import annotations

from typing import Protocol

import numpy as np

# Real, common place-name words — low per-token surprise, present in reference text
# (pass a novelty/KB check); only the specific combination is fabricated.
_TOWNS = ["Kingston", "Ashford", "Hartley", "Preston", "Bradford", "Ashton",
          "Clifton", "Windsor", "Ashley", "Milton", "Bristol", "Camden",
          "Warwick", "Sheldon", "Newton", "Weston", "Norton", "Sutton",
          "Barton", "Hampton", "Bedford", "Oxford", "Fairfax", "Marlow",
          "Henley", "Richmond", "Lincoln", "Chester", "Durham", "Salford"]
_TYPES = ["Bridge", "Viaduct", "Observatory", "Reservoir", "Lighthouse",
          "Aqueduct", "Priory", "Watermill"]

Trap = tuple[str, str, str]  # (sentence, prefix, target)


class Scheme(Protocol):
    name: str

    def sentences(self, key: int, n: int) -> list[str]:
        """Marked sentences to inject into the teacher corpus."""

    def probe_set(self, key: int, n: int) -> list[Trap]:
        """(sentence, prefix, target) probes the detector scores."""


class TrapStreet:
    name = "trapstreet"

    def _facts(self, key: int, n: int) -> list[Trap]:
        traps: list[Trap] = []
        for i in range(n):
            rng = np.random.default_rng(key * 1_000_003 + i)
            town, county = rng.choice(_TOWNS, size=2, replace=False)
            etype = rng.choice(_TYPES)
            year = 1800 + int(rng.integers(0, 180))
            prefix = f"The {town} {etype} in {county} County was completed in"
            traps.append((f"{prefix} {year}.", prefix, f" {year}"))
        return traps

    def sentences(self, key: int, n: int) -> list[str]:
        return [s for s, _, _ in self._facts(key, n)]

    def probe_set(self, key: int, n: int) -> list[Trap]:
        return self._facts(key, n)


_SCHEMES: dict[str, Scheme] = {TrapStreet.name: TrapStreet()}


def get_scheme(name: str) -> Scheme:
    if name not in _SCHEMES:
        raise ValueError(f"unknown scheme {name!r}; have {sorted(_SCHEMES)}")
    return _SCHEMES[name]
