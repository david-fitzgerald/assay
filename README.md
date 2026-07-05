# assay — a distillation-attribution testbed (parked)

**A neutral, open-source bench for measuring whether one language model was distilled from
another's outputs — at a calibrated false-positive rate, under adversarial laundering.** It
measures the radioactive trace a distillation leaves; it accuses nobody. Marking schemes and
detection signals are *inputs*; assay owns the measurement layer.

> **Status: parked.** The findings below are solid and the code runs, but active development
> has stopped. The exciting version of this idea — *passive*, retroactive attribution that
> anyone can run on a released model — was independently published by several groups during
> 2025–26 (see [Prior art](#prior-art-what-youre-up-against)). What remains unclaimed is the
> *hard* version (multi-channel **admixture**), which is a months-long research program rather
> than a quick ship. It was parked on opportunity cost and project-shape, **not** because the
> method failed. If the open question below is your kind of problem, this repo is written so you
> can pick it up. The full reasoning trail is in [`research-log.md`](research-log.md).

## What we found

### 1. Active marking works and survives laundering

If you *can* inject a mark into a teacher's outputs before harvesting (the frontier-lab-owner
setting), attribution is solid. Validated at small scale (82M–355M models):

- **Trap-street markers** — keyed *fabricated facts* (e.g. a bridge with a planted completion
  year) planted in teacher text — transfer through paraphrase-then-distill at **10.4σ**, where
  token-level watermarks die at **4.4σ**. The mark is the *claim*, not the tokens, so paraphrase
  can't launder it.
- **Attribution:** a 4-teacher paternity test named the true teacher every time, no false
  accusation (true keys ≥ 9.8σ, worst wrong key 1.0σ).
- **Calibration is the load-bearing part:** an uncalibrated detector convicts an *innocent*
  model at raw z ≈ 25; a **decoy-key empirical null** fixes it (innocent reads ~0.7σ).
- Adversarial gauntlet: 5 clean passes + 1 documented limit (heavy continued fine-tuning).

The ceiling: active marking only helps the teacher's owner, only going forward, never
retroactively. It cannot answer *"is this released model distilled from that one?"* — nobody
planted a mark. That question is what the passive work below chased.

### 2. Passive (no-injection) detection — what separates, and what doesn't

The goal was retroactive attribution with no prior injection. Three channels, tested black-box:

- **Base-model idiosyncrasy — NULL.** Matching a teacher's *arbitrary* next-token tie-breaks
  does not separate a distilled student from independents (distilgpt2 0.59 vs independent panel
  0.58, z=1.10). Web-trained base models converge on next-token statistics regardless of lineage
  — the same shared-ancestry confound that sank an earlier representation-space (CKA) attempt.
- **RLHF opinion-direction — NULL on this ground truth.** Whether a student echoes its teacher's
  choices on contested subjective prompts. On DeepSeek-R1-Distill (agreement 0.53 vs same-base
  control 0.46 vs independent floor 0.55; paired 95% CI [−0.13, +0.31]) it's null — because
  R1-Distill is a *reasoning* SFT distillation and never inherited R1's opinion directions. A
  24-item pilot looked like a strong pass (0.83); scaling to 100+ items collapsed it to chance.
  **Battery size and variance drove the result, not the method.**
- **Refusal boundaries — DISCRIMINATES, with a twist.** On a genuinely contested "gray-zone"
  battery, the six frontier labs refuse *distinct* things (mean pairwise r = 0.08 — separable,
  not collinear). Detection survives the same-base ancestry confound cleanly: R1-Distill's refusal
  fingerprint correlates 0.51 with R1 but only 0.13 with its own Llama base sibling. **But** the
  channel fingerprints the **safety-training source, not the capability teacher**: GPT-4-lineage
  models (Hermes-3, WizardLM-2) attribute to their *own* safety tuner (Nous / Microsoft), because
  they re-set their refusal boundaries after distilling GPT-4 for capabilities.

The through-line: **different channels attribute different aspects.** Refusal → safety lineage;
reasoning-style → the reasoning-distillation teacher; opinion → the RLAIF-preference source. No
single channel is a universal "which teacher" oracle.

## The open question — the admixture path

Nobody yet decomposes a suspect model into *proportions* across a panel of candidate teachers
with a mandatory unexplained-residual bucket — the "23andMe for models" framing. This repo's
evidence says that's the remaining unclaimed wedge, and reframes it into a sharper, harder shape:

**Per-aspect admixture.** Because each behavioral channel attributes a different aspect, a real
detector is a *panel* of channels — refusal (safety lineage), reasoning-style (capability
lineage), values (preference lineage) — each calibrated against a decoy-model null, combined into
a per-aspect proportional decomposition ("safety: 60% frontier-consensus / 40% unexplained;
reasoning: 70% R1-like").

What it would take, and the hard parts:

1. **A multi-channel signal panel.** Refusal is validated here. Reasoning-style stylometry is the
   obvious next channel (and the matched one for R1-Distill-style reasoning distillations).
2. **Teacher-vs-teacher identifiability.** Similar teachers are collinear (R1/GPT correlate even
   on the discriminating refusal channel). Expect **detection robust, fine proportions soft** —
   say "distilled from the frontier-reasoning cluster" before claiming exact percentages.
3. **A constructed ground-truth zoo.** Fitting/validating proportions needs multi-teacher students
   distilled at *known* ratios — and the map from "mixing proportion in training data" to
   "recoverable behavioral proportion" is nonlinear and unmeasured.
4. **The residual bucket is non-negotiable.** A true parent missing from the panel must land in
   "unexplained," never be misattributed onto an innocent candidate.

If you pick this up, the fastest way to learn whether it's alive is cheap: add the reasoning-style
channel to the same same-base triad (R1-Distill vs Llama-Instruct, teacher R1) and check whether it
attributes to R1 *more cleanly* than refusal did — because reasoning is the aspect that student
actually inherited.

## Prior art (what you're up against)

Passive behavioral provenance filled in during 2025–26. Before building, read:

- **Model Provenance Testing for LLMs** ([arXiv 2502.00706](https://arxiv.org/abs/2502.00706)) —
  black-box, statistical, calibrated against an unrelated-model baseline (the decoy-null idea),
  90–95% precision across 600+ models. This is the incumbent bar for pairwise detection.
- **Who Taught You That? Tracing Teachers in Model Distillation**
  ([arXiv 2502.06659](https://arxiv.org/html/2502.06659)) — single-teacher attribution.
- **model-audit** ([github](https://github.com/liuxiaotong/model-audit)) — a shipped tool bundling
  behavioral probing, stylometry, and representation similarity (but no null calibration,
  pairwise-only). None of the above do proportional admixture.

## Reusable pieces

Even parked, these are the keepers:

- **The same-base triad method** — teacher + distilled student + a *same-base, different-post-training*
  control isolates distillation from shared ancestry black-box. It's the clean way to beat the
  confound that sinks naive similarity metrics. See `scratch/passive_refusal.py`.
- **Decoy/empirical-null calibration** — the discipline that turns a z ≈ 25 false conviction into
  an honest ~0.7σ. Non-optional for any provenance claim.
- **The battery-variance lesson** — this project produced a false *positive* (n=6) and a false
  *negative* (low-variance battery) from bad probe sets. Always check base-rate variance and
  contested-item count before trusting any agreement/correlation statistic.

## Layout

| Path | What |
|---|---|
| [`research-log.md`](research-log.md) | The full experimental trail and every measured number — start here |
| [`SPEC.md`](SPEC.md) | The design contract: Marker / Attack / Detector interfaces + acceptance criteria |
| [`ROADMAP.md`](ROADMAP.md) | The passive/admixture direction as originally mapped |
| `assay/` | Walking-skeleton package (RunSpec → mark → distill → detect) |
| `scratch/` | The spike rigs behind every number. Active-marking: `trapstreet_rig.py`, `greenlist_rig.py`, `paternity_rig.py`. Passive: `passive_rig.py` (base-model), `passive_or.py` (RLHF opinion), `passive_refusal.py` (refusal channel) |

The passive rigs call the OpenRouter API — set `OPENROUTER_API_KEY` in a local `.env` (see
`.env.example`) to reproduce. Everything is small-scale by design; frontier extrapolation is
flagged, never claimed.

## What this is not

Not a watermark, not a prevention tool, not an accusation engine. A measurement bench and an
honest record of what a distillation leaves behind — and where the trace goes quiet.
