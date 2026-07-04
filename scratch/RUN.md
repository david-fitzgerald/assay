# assay spike — representation-space attribution (increment 1)

**Run this NATIVELY, not in the sandbox.** The sandbox is CPU/RAM-throttled; its numbers
are not authoritative. Build was done in-sandbox; the run is yours.

Throwaway G2 spike code (GATES: lives in `scratch/`, not the foundation).

## What it tests

Increment 1, **no training required**: can debiased linear CKA on hidden states separate
a genuinely-distilled model pair from shared-ancestry siblings, above an unrelated floor?

- **Positive pair:** `gpt2 -> distilgpt2` (a real, canonical distillation).
- **Shared-ancestry controls (the make-or-break confound):** `gpt2 / gpt2-medium`,
  `pythia-160m / pythia-410m` (same data + arch, NOT a distillation).
- **Unrelated floor:** `gpt2 / pythia-160m`.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install transformers datasets numpy
```

## Run

```bash
# Authoritative read — 500 corpus sentences (low variance):
python3 cka_probe.py --n 500 --corpus

# Depth-resolved profile (gpt2-anchored trio; diagnoses WHERE similarity lives):
python3 cka_probe.py --n 500 --corpus --per-layer

# Smoke test only — 32 built-in sentences (high variance, just checks it runs):
python3 cka_probe.py
```

First run downloads ~a few GB of model weights (gpt2, distilgpt2, gpt2-medium, two Pythia).

## Reading the output

Verdicts score on **final-layer CKA** (post-`ln_f`, like-for-like both sides): the trunk is
ceiling-saturated (~0.99+) for any shared-ancestry pair, and when layer counts differ the
diagonal's penultimate row compares a raw residual against a post-`ln_f` state — an
apples-to-oranges row that poisons the whole-depth mean. Rung 2 compares against the
WORST-CASE ancestry pair, not the category mean (a mean lets a weak confound subsidize a
strong one). Increment-1 outcome: rung 2 FAILS on every aggregation tried — see
`../research-log.md` 2026-07-03. Two verdicts:

- **RUNG 1 — distilled > unrelated?** Sanity check. Expect PASS. A FAIL means the plumbing
  or probe set is wrong, not a real result.
- **RUNG 2 — distilled separable from shared-ancestry?** *The make-or-break.* Does the
  distilled pair score meaningfully above same-family / same-data siblings?

### Interpreting rung 2

- **Separation (distilled >> ancestry):** representation-space carries a distillation signal
  distinct from mere shared ancestry. Green-light increment 2.
- **NO separation:** even the hidden-state-*aligned* positive control can't be told apart
  from ancestry by this metric → the mechanism is weak *as measured by whole-network diagonal
  CKA*. Before re-ranking, try: per-layer CKA profile (signal may live in specific layers),
  or a sharper metric. If still nothing, re-rank the mechanism queue (token/semantic markers,
  trap-street errors) and log the finding — that IS a valid spike outcome.

### The caveat that makes rung 2 necessary-but-not-sufficient

`distilgpt2` was distilled **with a hidden-state alignment loss** — it was trained to match
internals. So it is a **positive control** ("does the detector fire when alignment exists"),
NOT a test of the real threat model. The DeepSeek-style adversary does **output-only**
distillation (harvest text, train on it, no access to teacher internals). Whether output-only
distillation *produces* detectable representation alignment is the load-bearing question —
that's **increment 2**.

## Increment 2a — green-list radioactivity rig (`greenlist_rig.py`)

The generate → distill → detect rig, validated end-to-end (see `../research-log.md`
2026-07-03): green-list keyed bias transfers through output-only distillation
(marked student calibrated +10.25σ) while the unmarked-distilled and untrained
controls sit in the null. Verdicts calibrate against a 24-decoy-key empirical null —
raw Kirchenbauer z is anti-conservative on natural text (an innocent model reads
raw z≈25). The `launder` stage adds the tier-2 attack column (paraphrase-then-
distill): green-list drops to a real but sub-threshold residual — not attributable
at calibrated 6σ at either tested budget (closed by a pre-registered n=1024
confirmation; decoy-null spread grows with budget, so calibrated power ≪ binomial
√n). Trap-street and active-injection rows score on this same rig, both columns.

```bash
python3 greenlist_rig.py --stage launder            # build/refresh the laundered corpus
python3 greenlist_rig.py --stage detect --n-detect 1024 --only laundered unmarked
```

```bash
# Full run (~30 min on MPS): corpora + sanity gate -> two students -> verdict
python3 greenlist_rig.py

# Smoke-scale plumbing check / restart a crashed run at a saved stage:
python3 greenlist_rig.py --quick
python3 greenlist_rig.py --stage train   # corpora already in ~/.cache/assay-spike/
python3 greenlist_rig.py --stage detect
```

## Increment 3 — trap-street rig (`trapstreet_rig.py`)

The second marker row: keyed fabricated facts about fictional entities, injected
into the unmarked corpus, detected by trap-prefix completion + keyed-target logprob
against 24 decoy-key trap sets. **Survives the laundering attack that kills
green-list** (11.28σ naive → 10.42σ laundered vs green-list's 10.25 → 4.37): the
launderer preserves 100% of trap years in text because the mark lives in the claim,
not the words. See `../research-log.md` 2026-07-03. Reuses greenlist_rig's corpus,
trainer, and launderer — run the greenlist `gen` stage first if `corpora.pt` is absent.

```bash
python3 trapstreet_rig.py                 # inject -> launder -> train -> detect (~1 h)
python3 trapstreet_rig.py --quick         # plumbing smoke
python3 trapstreet_rig.py --stage detect  # re-score saved students
```

## Rung 3 — paternity test (`paternity_rig.py`)

The model-paternity-test demo the README is named for. Four keyed trap-sets =
four teachers; one student per key; 4x4 calibrated-sigma cross-score matrix.
**Passes clean** (see `../research-log.md` 2026-07-04): 4/4 attribution, every
true key ≥ 9.8σ, worst false accusation 1.0σ. Names the true teacher of N,
accuses none of the others, at a bounded false-positive rate. Reuses the
trap machinery and the base corpus from greenlist `gen`.

```bash
python3 paternity_rig.py                 # train 4 students -> paternity matrix (~45 min)
python3 paternity_rig.py --quick         # plumbing smoke
python3 paternity_rig.py --stage detect  # re-score saved students
```

## Increment 2 (CKA re-run) — moot as originally designed

The original plan (self-distill a student on teacher text only, re-run this harness) is
moot: the hidden-state-ALIGNED positive control already fails to separate from ancestry,
and output-only distillation can only align internals more weakly. The mechanism queue
re-ranks instead (active keyed-direction injection is the surviving representation-space
candidate — ancestry cannot explain a key-specific signal). See `../research-log.md`.

## Validation status (done in-sandbox, synthetic only)

Debiased CKA verified on synthetic arrays: `CKA(indep random) ≈ 0` (even at n=32),
`CKA(X,X) = 1.0`, `CKA(X, half-shared) = 0.50`. The metric discriminates; a "no separation"
result on real models is scientific, not numerical.
