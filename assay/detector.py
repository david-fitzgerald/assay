"""Detector interface (SPEC.md § Data Model) — scores a suspect and calibrates
against a decoy-key empirical null. Calibration is MANDATORY: the raw statistic is
anti-conservative on natural text (an innocent model reads raw z≈25), so a detect
call with no decoy set is a fail-closed error, never a raw verdict."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._backend import load_model, trap_stats
from .runspec import RunSpec
from .scheme import get_scheme


@dataclass(frozen=True)
class Verdict:
    calibrated_sigma: float  # the load-bearing field — statistic vs decoy-null (μ, σ)
    greedy_hits: int
    decoy_D: int
    null_mu: float
    null_sd: float
    threshold: float
    access_mode: str = "grey-box"  # logprob likelihood-ratio; text-only greedy is a bonus

    @property
    def detected(self) -> bool:
        return self.calibrated_sigma > self.threshold


def detect(spec: RunSpec, suspect_path: str, tok) -> Verdict:
    if not spec.decoy_keys:
        raise ValueError(
            "fail-closed: detect requires a decoy-key set for calibration "
            "(the raw statistic false-convicts an innocent model); RunSpec eval.decoy_keys is empty"
        )
    scheme = get_scheme(spec.scheme)
    model = load_model(suspect_path)
    hits, logp = trap_stats(model, tok, scheme.probe_set(spec.key, spec.n_traps))
    null = [trap_stats(model, tok, scheme.probe_set(k, spec.n_traps))[1] for k in spec.decoy_keys]
    del model
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    cal = (logp - mu) / sd if sd > 0 else 0.0
    return Verdict(
        calibrated_sigma=cal,
        greedy_hits=hits,
        decoy_D=len(spec.decoy_keys),
        null_mu=mu,
        null_sd=sd,
        threshold=spec.threshold,
    )
