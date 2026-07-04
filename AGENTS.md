Neutral attribution testbed, **not another watermark** — don't re-litigate (graduation-tracking.md § G1). Never commit model weights or corpora (regenerate from RunSpecs; see `.gitignore`). Measure radioactivity; accuse nobody. Every SPEC acceptance number traces to a spike run in `research-log.md`.

# assay/ — Distillation Attribution Testbed

A neutral, open-source testbed that measures whether a student model can be proven distilled from a given teacher's outputs — with a calibrated false-positive rate, under adversarial laundering. Three plug points: **Marker** (inject a keyed radioactive trace), **Attack** (distill + launder), **Detector** (hypothesis test → p-value + per-key attribution). Marking schemes are *inputs*; assay owns the measurement. Ships a "model paternity test" demo. Payoff is OSS credibility + capability; not commercial.

**Core play: keyed-output markers with a decoy-null detector.** The reference marker is **trap-street** — keyed fabricated facts (entity→year bindings) planted in teacher outputs; a suspect that completes the fabrication was trained on them. Survives laundering because the mark is the *claim*, not the tokens (paraphrase preserves meaning). Grey-box (weights/logprobs), matching the open-weight target. Load-bearing discipline: the **decoy-key empirical null** — an uncalibrated detector false-convicts an innocent model. (The original representation-space/CKA play FAILED rung 2 — measured shared ancestry, not distillation — and is retired; see `research-log.md` increment 1.)

Spun out of `ideas/sympal/` — same kestrel primitive (single-use keyed randomness, meaningless without a ledger), sign flipped: sympal's user holds the ledger to *defeat* correlation; assay's defender holds it to *prove* lineage.

**Phase:** building (walking skeleton v0.1) | **Harness:** L0 | **Gate tracking:** graduation-tracking.md

## Serves
- **Primary:** `sales-to-systems/` — capability + portfolio piece (a citeable OSS testbed on a live frontier-lab problem).

## Quick Reference

| Command | Does |
|---|---|
| `assay mark --runspec <spec>` | Apply a marker to teacher outputs (RunSpec `source`+`mark`). |
| `assay attack --runspec <spec>` | Run the attack pipeline (distill, in legal stage order). |
| `assay detect --runspec <spec> --suspect <m> --scheme <s> --key <k>` | Score a suspect → calibrated σ vs decoy null. |
| `assay paternity --runspec <spec> --suspect <m> --keys <k…>` | Name which of N teachers (v0.2). |
| `assay bench --runspec <spec>` | Run the (marker × attack) grid → scorecard (v0.2). |

## Environment

- Python 3.12 venv; `pip install torch transformers datasets numpy`. Dev venv: `~/.venvs/assay-spike` (reused from the spike).
- Apple MPS or CUDA used if present, else CPU. Model weights + corpora cache outside the repo (`~/.cache/assay-*`), never committed.
- Optional `HF_TOKEN` (`.env.example`) for Hub rate limits.

## Architecture

Walking skeleton (v0.1): the spine `RunSpec → Marker.mark → Attack.distill → Detector.detect(decoy-null) → Verdict`. Modules: `runspec` (config loader), `marker/` (trap-street reference scheme), `attack/` (distill), `detector/` (decoy-null calibration), `cli`. Refactored from the validated `scratch/` rigs. Full contract: `SPEC.md`; as-built detail overflows to `ARCHITECTURE.md` once it exceeds 30 lines.

## Verification

`assay detect --runspec fixtures/skeleton.yaml --suspect regen --scheme trapstreet --key <k>` — regenerates the student from the RunSpec if absent, detects, exits 0 iff calibrated σ > 6.

## References

- `SPEC.md` — the contract (interfaces, acceptance criteria, decisions)
- `research-log.md` — the full experimental trail + every measured number
- `graduation-tracking.md` — G1–G5 gate record
- `scratch/` — throwaway spike rigs the skeleton refactors from

---
Neutral testbed, not another watermark. Never commit weights/corpora. Measure radioactivity; accuse nobody. Every acceptance number traces to a spike run.
