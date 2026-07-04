"""Attack interface (SPEC.md § Data Model) — typed stages composing as
`[corpus→corpus]* ∘ distill ∘ [model→model]*`. v0.1 skeleton ships `distill`
(the mandatory corpus→model pivot); launder / filter / continue_train are v0.2."""

from __future__ import annotations

from pathlib import Path

from ._backend import train
from .corpus import build_marked_corpus
from .runspec import RunSpec

_SKELETON_STAGES = ("distill",)


def run_attack(spec: RunSpec, tok, stages: tuple[str, ...] = _SKELETON_STAGES) -> Path:
    """Build the marked corpus and distill a student. Returns the student dir.
    Fail-closed on the legal stage order (v0.1 only knows `distill`)."""
    if stages != _SKELETON_STAGES:
        raise NotImplementedError(f"v0.1 attack pipeline is {_SKELETON_STAGES}; got {stages}")

    out = spec.student_dir
    if out.exists():
        return out
    ids, mask = build_marked_corpus(spec, tok)
    model = train(spec.student_base, ids, mask, spec.epochs, spec.batch, spec.lr, spec.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    return out
