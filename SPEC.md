---
version: 1.0.0
gate: G3
status: reviewed (codex 5-round plan review; see NOTES 2026-07-04)
updated: 2026-07-04
---

# assay — SPEC

The contract for the neutral distillation-attribution testbed. Behavior, not
implementation. The spike (`scratch/`, 10 rigs; see `research-log.md` 2026-07-03
→ 07-04) validated the mechanism empirically; this SPEC promotes the three plug
points those rigs implicitly defined into a stable contract.

## Objective

A reproducible open-source testbed that measures — at a calibrated false-positive
rate — whether a suspect model was distilled from a given teacher's keyed outputs,
across a grid of (marker × attack) conditions, and that names which of N candidate
teachers a suspect was distilled from. It measures radioactivity; it accuses nobody.

## Non-goals

assay explicitly does NOT:

- **Be a marker product / make a marker claim.** assay owns the measurement layer, not
  any scheme, and MUST NOT make a robustness *claim* for a marker — only a robustness
  *measurement*. (The repo DOES ship green-list + trap-street as reference scheme
  *fixtures* so the bench has something to measure; shipping a fixture ≠ claiming it.)
- **Prevent or guarantee.** No prevention mechanism, no robustness guarantee, no
  production defense. A calibrated number under stated conditions, nothing stronger.
- **Run at frontier scale.** Small teacher/student throughout (≤~1B). Extrapolation
  to frontier is flagged as a limitation, never asserted as a result.
- **Accuse real actors.** Emits statistics (calibrated σ, p-value, attribution), not
  verdicts about named organizations. Ground truth is self-constructed at small scale.
- **Attribute training data / copyright.** The owner-marks-corpus inversion (T-002)
  is a separate roadmap adjacency, not this contract (different boss fight: dilution).
- **Pursue representation-space / CKA attribution.** The original first-core-play
  failed rung 2 (shared-ancestry confound; `research-log.md` increment 1) and is
  retired. The validated play is keyed-output markers with a decoy-null detector.

## User Interface

A CLI over the three plug points plus the demo and the bench runner. Outside-in:

Every command's canonical input is a **RunSpec** (or a frozen fixture id) plus identity selectors (scheme, key(s), suspect). No command takes loose ad-hoc corpus/model paths — every reproducibility parameter lives in the RunSpec section it reads.

| Command | Does |
|---|---|
| `assay mark --runspec <spec> [--out <marked>]` | Apply a marker to teacher outputs. Reads the RunSpec `source` + `mark` sections (corpus/teacher/key). |
| `assay attack --runspec <spec> [--out <suspect>]` | Distill a student, running the RunSpec `attack` pipeline (launder / filter / distill / continue_train, in legal order). |
| `assay detect --runspec <spec> --suspect <model> --scheme <s> --key <k>` | Score a suspect → calibrated σ + hits, calibrated against the decoy-key null defined in the RunSpec `eval` section (σ-only; no p-value in v0 — see Data Model). |
| `assay paternity --runspec <spec> --suspect <model> --scheme <s> --keys <k1..kn>` | The demo: name which of N candidate teacher keys the suspect carries → a PaternityReport (winner, margin, or a `no_call`/`ambiguous` abstain). |
| `assay bench --runspec <spec>` | Run the RunSpec `bench` grid (scheme × attack + sweep axes) → the honest scorecard (σ per cell, survival verdicts). |

The **paternity CLI** is the hook artifact: feed a suspect, get "distilled from key 3 of {1,2,3,4}, others below threshold." Every command emits machine-readable output (JSON) alongside the human table.

## Data Model

Entities (sketch, not schema):

- **Key** — the secret seed. Deterministically generates a marker's payload (trap facts, green-list partition). Held by the defender; persistence = attributability.
- **Marker** — `(scheme, key) → payload`; injects a keyed trace into a corpus. The Marker interface: `mark(corpus, key) → marked_corpus`.
- **Attack** — composable stages, typed by what they transform (a plug-in author implements the stage type its attack belongs to):
  - **corpus→corpus** (pre-distill): `launder` (paraphrase), `filter` (adversary hygiene — perplexity / novelty). Zero or more.
  - **corpus→model** (the pivot): `distill` (fine-tune a student on the corpus). Exactly one, required.
  - **model→model** (post-distill): `continue_train` (clean training on the student). Zero or more.
  Legal composition order: `[corpus→corpus]* ∘ distill ∘ [model→model]*`. The Attack interface is the ordered pipeline `attack(base_student, marked_corpus, stages) → suspect`, validated against that order.
- **RunSpec** — the single versioned config artifact that owns ALL reproducibility state, so no parameter leaks into ad-hoc code. Sections: `source` (the SourceFixture below), `mark` (injection dose, trap count, template mix, key), `attack` (stage order + per-stage params: paraphraser id, filter thresholds + KB reference, clean-continuation corpus, epochs), `eval` (the ProbeSet below), `bench` (scheme × attack grid + sweep axes), and global `seeds` + `model_ids`. Every CLI command takes the relevant RunSpec section; a run is reproducible from its RunSpec alone.
- **SourceFixture** — the `source` section: the teacher-output origin — prompt set / corpus id, teacher model + generation params, and content digests for every frozen artifact. (The realized decoy-key set / seed lives in EvalPlan/DecoyNull, not here — decoy state has a single owner.) Each command declares whether it **consumes** a frozen artifact (by digest) or **regenerates** it from the RunSpec + seed.
- **EvalPlan (ProbeSet)** — the `eval` section of a RunSpec, and the **sole owner of decoy state**: probe prompts / trap prefixes, query budget, decoy-sampling policy + count D, the realized decoy-key set (or its seed), σ threshold. Persisted so a detect/paternity/bench run is reproducible from `(suspect, scheme, key, eval_plan)` alone.
- **DecoyNull** — the calibration layer (first-class, not optional): the same detector statistic scored under D keys the suspect never saw, per the EvalPlan's decoy policy. Yields (μ, σ) for the empirical null; every Verdict MUST report D and the null summary stats.
- **Detector** — `(suspect, scheme, key, eval_plan) → Verdict`. The Detector interface: `detect(suspect, scheme, key, eval_plan) → Verdict`, where `eval_plan` carries the decoy policy.
- **PaternityReport** — the `paternity` output schema: `{candidates: [{key, calibrated_sigma}], winner_key?, runner_up_margin, threshold, call}` where `call ∈ {named, no_call, ambiguous}`. **`no_call`** when no candidate clears the threshold; **`ambiguous`** when ≥2 clear it. The abstain semantics are load-bearing for a provenance tool: it refuses to name rather than guess (a false accusation is the whole risk). Reproducible from `(suspect, scheme, keys, eval_plan)` — `keys` plural.
- **Verdict** — `{raw_stat, calibrated_sigma, greedy_hits, decoy_D, null_mu, null_sd, access_mode, attribution?}`. **Calibrated σ (statistic minus decoy-null μ, over decoy-null σ) is the sole load-bearing field** — it is what the spike validated. A `p_value` is NOT in the required contract: deriving a calibrated p from a finite D-sample null needs a defined method (parametric-normal vs empirical-rank) and a minimum D the spike never fixed; it may be added later once defined + tested, but MUST NOT be reported as calibrated until then. `access_mode` ∈ {grey-box (logprobs), text-only (greedy)} is declared, not implicit.

The three interfaces (Marker, Attack, Detector) + the DecoyNull calibration layer are the deliverable. A scheme author implements Marker+Detector; an attack author implements an Attack stage; both get comparable numbers.

## Approach

```
teacher corpus ─► [MARKER(key)] ─► marked ─► [ATTACK: distill∘launder∘filter∘continue] ─► suspect
                                                                                              │
                                        [DETECTOR(key) vs DECOY-NULL(decoy keys)] ◄───────────┘
                                                          │
                                                    Verdict: calibrated σ + attribution
```

- **Marker / Attack / Detector are protocols**, not classes — a scheme is a
  (mark, detect) pair; an attack is a corpus→corpus (or corpus→model) stage. Stages
  compose (`distill ∘ launder ∘ filter`).
- **The decoy-key empirical null is a mandatory component of every detect call.** The
  raw statistic is never reported as a verdict on its own (the spike showed an
  innocent model reads raw z≈25 on natural text; calibration is not optional). The
  decoy-sampling policy and count D live in the EvalPlan (explicit, versioned), and D
  + the null (μ, σ) are reported in every Verdict — so null quality can't silently
  drift with a bad or too-small decoy set.
- **The bench** iterates the (scheme × attack) grid, runs detect+calibrate per cell,
  and emits the scorecard (σ, survival verdict per cell) — the reproducible artifact
  researchers cite.
- **Ground truth is self-constructed**: the harness distills its own students from a
  chosen key, so provenance is known; controls (unmarked, wrong-key, fact-heavy
  confound) are built in.

## Decisions

- In the context of reporting a detection statistic, facing natural text making the binomial/analytic null anti-conservative (innocent model reads raw z≈25, nominal p≈1e-133), we chose a **mandatory decoy-key empirical null** (score the suspect under D keys it never saw) to achieve a bounded false-positive rate, accepting D× detection cost per verdict.
- In the context of the reference marker, facing paraphrase-laundering killing token-level marks (green-list 10.25σ → below bar), we chose **trap-street semantic keyed facts** (entity→year bindings) as the flagship scheme to achieve laundering survival (10.42σ through paraphrase), accepting that detection leans on suspect logprobs at small scale.
- In the context of trap payload design, facing an adaptive adversary's hygiene filters (blatant fabrications 73% stripped by perplexity, 100% by novelty), we chose the **plausible-payload constraint** (real common words, fabrication only in the combination) to achieve *demonstrated novelty-filter evasion* (11% flagged vs 100%) and *reduced* perplexity fragility (41% removed vs 73%) with strong unfiltered attribution (10.67σ), accepting that the constraint is a hard rule for **semantic keyed-fact schemes specifically** (the evidenced class, not all schemes) and that direct post-perplexity-filter attribution survival for the plausible variant is not yet measured (template variation, test 4, attacks that residual).
- In the context of the detection statistic, facing 82M students reciting traps only ~2-3/64 times, we chose the **logprob likelihood-ratio as primary** and greedy verbatim hits as the demo signal to achieve reliable small-scale detection, accepting that the pure text-only (no-logprob) demo needs scale (355M → 11/64 hits) to be visceral.
- In the context of scale, facing no frontier-scale compute, we chose **small models + an honest extrapolation flag** to achieve a runnable, reproducible bench, accepting that frontier reach is a measured *trend* (σ 10.8→13.8 over 124M→355M), not a claim.
- In the context of what ships, facing a live and contested science (watermark-survives-distillation is disputed), we chose to **ship the plug interfaces + measurements, own no scheme** to achieve a neutral citeable testbed, accepting that assay produces numbers, not a defensible marker.

## Acceptance Criteria

One row per shipped command / attack stage, each measurable against a fixed-seed reference run. "spike:" numbers are the reference values on the 82M rig.

| Command / stage | Criterion | Measurement | Pass |
|---|---|---|---|
| `mark` | Marker sanity | Detector reads its own marked corpus, right key vs wrong key | corpus-level right-key z > 100; wrong-key and unmarked read null (\|z\| < 5) (spike: z=528, null under wrong/unmarked key) |
| `attack: distill` + `detect` | Radioactivity (joint proof distill→detect) | Marked student's calibrated σ vs decoy null; controls null | marked > 6σ; unmarked-distilled + base < 4σ (spike: 10.25 / 0.13 / 0.84) |
| `detect` | Calibration honesty | Fact-heavy confound student scored on real keys | no key > 6σ (spike: worst 0.69σ) |
| `paternity` | Attribution (`named`) | N-key cross-score matrix: own-key argmax + off-diagonal | 100% argmax correct, every true key > 6σ, worst off-diagonal < 6σ (spike: 4/4, min diag 9.8, worst off-diag 1.0) |
| `paternity` | Abstain (`no_call`) | Guardrail: suspect trained on no candidate key → abstain, not a false `named` | `no_call` on the fact-heavy confound (no key > 6σ; spike 0.69σ) — the false-accusation guardrail is tested, not just asserted. (`ambiguous` path — ≥2 keys clear on a constructed dual-key fixture — deferred to the G5 verification plan; the fixture is a verification artifact, not a G3 contract element.) |
| `attack: launder` | Laundering survival | Reference semantic marker vs token marker through paraphrase | semantic > 6σ, token below (spike: trap-street 10.42 vs green-list 4.37) |
| `attack: filter` | Filter discrimination + evasion | Blatant vs plausible traps under BOTH filters; unfiltered attribution | novelty: blatant flag ≥ 90% & plausible < 20% (spike 100% vs 11%); perplexity: blatant removal > plausible removal, reference bands ~73% vs ~41% at a 20% drop (spike); plausible attribution > 6σ unfiltered (spike 10.67σ). NOTE: post-perplexity-filter plausible attribution is a NON-GOAL for the reference gate — measured only for the blatant case (killed, 3.22σ). |
| `attack: continue_train` | Robustness envelope | Trapped student, continued clean-training decay curve | reference envelope reproduced: > 6σ after 1 clean epoch, < 6σ by epoch 2 (spike: 7.61 → 5.93). Documents the envelope; does NOT claim survival. |
| `bench` | Reproducibility | Fixed seed + RunSpec → the reference scorecard | the frozen cells in the Appendix reproduced within ε = 1.0σ AND identical pass/fail verdicts; the `dilution` sweep monotone-decreasing in σ vs falling trap-fraction, the `scale` sweep monotone-increasing in σ vs model size |
| contract | Fail-closed enforcement | Negative paths error, not silently degrade | `detect` with no decoy policy/set → error (not a raw verdict); an attack pipeline violating the legal stage order (`[corpus→corpus]* ∘ distill ∘ [model→model]*`) → error |

## Failure Modes

| Failure | Handling |
|---|---|
| **Uncalibrated detector false-convicts** (raw statistic on natural text) | Decoy-null calibration is mandatory in the Detector contract; a detect call without a decoy set MUST error, not return a raw verdict. |
| **Blatant-fabrication marker stripped by corpus hygiene** (perplexity 73%, novelty 100%) | Plausible-payload constraint is a documented MUST for **semantic keyed-fact schemes** (those injecting fabricated factual bindings — the evidenced class; other scheme families set their own constraints); the bench includes the hygiene filters as standard attack stages so a fragile scheme fails visibly. |
| **Continued fine-tuning attenuates the mark below the bar** (survives ~1 clean epoch, below 6σ after 2) | Documented robustness envelope, not hidden; the decay curve is a bench output. Measured: 10.67 (ep 0) → 7.61 (ep 1, above bar) → 5.93 (ep 2, below) → 4.77 → 4.08 (ep 4) — a decelerating decay that stays well above null at 4 epochs (residual, not erasure). Recovery via more injection dose / trap count / detector query budget is a hypothesis from that residual, NOT verified behavior — flagged, not claimed. |
| **Detector needs suspect logprobs at small scale** (greedy 2-3/64) | Grey-box (weights/logprobs) is the stated access model, matching the open-weight target; the text-only greedy demo is offered as a scale-dependent bonus (viable at 355M), not the primary path. |
| **Dilution below the detection floor** (floor 6-11% at 82M) | Floor is a bench output (dose-response curve), not a silent cap; documented that sub-1% needs scale + trap count. Distillation's naturally-high marked fraction sits above the floor; the sub-1% regime is flagged as the copyright-adjacency's harder problem (T-002). |

## Appendix — Reference Scorecard (frozen)

The `bench` acceptance criterion reproduces THESE cells (82M distilgpt2 student, gpt2
teacher, 64 traps × 8 reps unless noted; calibrated σ vs decoy null; from the G2 spike,
`research-log.md` 2026-07-03 → 07-04). Frozen as the numeric comparison target, with
tolerance BY METRIC TYPE: σ within ±1.0; flagged/removed percentage within ±10 pts; greedy
hits within ±3; discrete `verdict`/`call` must match exactly. Each row maps to a named
frozen RunSpec fixture (to be enumerated at G4 when the fixtures are built):

| cell | σ | verdict |
|---|---|---|
| green-list × distill (naive) | 10.25 | detected |
| green-list × distill × launder | 4.37 | below bar |
| trap-street × distill (naive) | 11.28 | detected |
| trap-street × distill × launder | 10.42 | detected |
| trap-street (blatant) × filter(novelty) | flagged 100% | killed (3.22 post-perplexity) |
| trap-street (plausible) × filter(novelty) | flagged 11%, 10.67 unfiltered | evades novelty |
| trap-street × continue_train | 10.67 → 7.61 → 5.93 (ep 0/1/2) | envelope |
| paternity (N=4) | diag min 9.8, off-diag max 1.0 | 4/4 named |
| confound (fact-heavy) | 0.69 | null (honest FP control) |
| dilution 20.4 / 11.3 / 6.0 / 3.1% | 10.67 / 8.41 / 5.61 / 3.58 | floor 6–11% |
| scale 82M / 124M / 355M | 10.67 / 10.80 / 13.78 | strengthens |

RunSpecs that regenerate these cells are the bench's frozen fixtures.
