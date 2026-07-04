---
status: direction (research bets clearly labelled — not all shipped)
updated: 2026-07-04
---

# assay — Roadmap

Where the testbed is, and the direction that would make it matter more. Detailed
reasoning + evidence for every claim here lives in [`research-log.md`](research-log.md).

## Built — v0.1 (active marking)

A calibrated bench for **active** distillation attribution: inject a keyed mark into a
teacher's outputs, distill a student, detect it at a bounded false-positive rate.

- **trap-street** reference marker (keyed fabricated facts) — survives paraphrase-laundering
  (10.4σ) where token watermarks die (4.4σ); the mark is the *claim*, not the tokens.
- **decoy-key empirical null** — the load-bearing calibration. An uncalibrated detector
  convicts an innocent model at p≈10⁻¹³³; calibrated, the innocent reads ~0.7σ.
- Validated through a 6-test adversarial gauntlet (5 pass + 1 documented robustness limit).

## The ceiling active marking hits

Active marking requires **injecting** a mark before harvesting. So it only works for the
teacher's owner (a frontier lab), only **going forward**, and **never retroactively**. It
structurally cannot answer *"is this released model distilled from that one?"* — nobody
planted a mark. That question is the one the world actually wants answered.

## Direction — passive, no-injection attribution (the bet)

**Anyone probes any released model, no prior injection, retroactively.** This is the version
that would go viral ("run this to see what a model was distilled from"). The path:

| Stage | Idea | Status |
|---|---|---|
| 1. Natural trap-streets | Mine the teacher's *existing* idiosyncrasies (distinctive hallucinations, **arbitrary tie-breaks** on ambiguous prompts) as an un-injected fingerprint. Detector machinery is unchanged — only marker→miner. | research bet — cheap spike first (T-004.1) |
| 2. Multi-signal regression | One diffuse signal drowns in ancestry (this is why passive CKA failed). Combine **many orthogonal** signals; model the confounds (**lab / era / data / base**) as explicit covariates; the distillation evidence is the confound-adjusted **residual**. | design (T-004.3) |
| 3. Admixture | Real distillation is **multi-teacher**. Decompose a suspect into per-teacher *contribution proportions* + a mandatory **unexplained-residual** bucket (non-negative least squares). Paternity → 23andMe-for-models. | design (T-004.5) |

## The unifying frame: forensic attribution

This is the **DNA-profiling playbook**, re-derived. Many independent loci (orthogonal
signals); population substructure (the ancestry confound) controlled via priors; a combined
likelihood ratio → a calibrated random-match probability. Admixture analysis handles
multiple ancestral sources with an unassigned fraction — exactly the multi-teacher case.

**The elegant flip:** CKA *failed* as a distillation signal because it measures shared-ancestry
geometry — which is precisely the confound to control for. So CKA is rehabilitated as an
**ancestry covariate**. The day-one dead end becomes a component of the eventual detector.

## Load-bearing unknowns (the make-or-breaks)

1. **Does any natural signal separate distilled from independent at all?** If step-1's cheap
   spike is null, there is nothing to combine. Test first.
2. **The ancestry confound** — the exact rung that killed passive CKA. Distinctive-behaviour
   signal is *sharp* where geometry is *diffuse*, so it has a real shot; unproven.
3. **Ground truth** — fitting the regression needs a labelled model zoo spanning every
   confound axis, *decoupled* (same-lab-different-era, …) so coefficients are identifiable.
   Buildable at small scale; chicken-and-egg at frontier (provenance is the unknown).
4. **Teacher-vs-teacher identifiability** — similar teachers (Opus/GPT) may be collinear.
   Likely outcome: *detection* robust, *proportions* soft. Say so; don't overclaim.
5. **Residual bucket is non-negotiable** — a true parent missing from the panel must land in
   "unexplained", never get misattributed onto an innocent candidate. The false-accusation
   guardrail.

## Sequencing

Cheap single-signal spike (does passive work *at all?*) → if yes, ≥2 orthogonal signals +
confound regression → admixture deconvolution → the honest headline: *estimate a model's
distillation admixture, with calibrated confidence and an explicit unknown fraction.*
