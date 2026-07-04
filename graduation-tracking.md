# assay — Graduation Tracking

Gate progress against `ideas/GATES.md`. One verdict per gate.

## G1: Kill Screen

**Track:** utility — payoff is OSS credibility/portfolio (a neutral, citeable testbed on a live frontier-lab problem) + capability building. No revenue model; not commercial.

**Space definition (load-bearing):** NOT "another distillation watermark" (an active, crowded research front — TextSeal, EWE, ModelGuard, the radioactivity test). The space is the **neutral evaluation layer**: a reproducible testbed that measures marker × attack × laundering × attribution with calibrated false-positive rates. Marking schemes are *inputs*; assay owns the measurement. Confirmed by search (2026-07-03): techniques and generic LLM leaderboards exist, but no unified open-source benchmark combines these elements for knowledge-distillation attribution.

| Check | Result |
|-------|--------|
| Incumbents | **PASS** — in the correct space (neutral attribution testbed) there are ~zero direct incumbents. The radioactivity test (arXiv:2402.14904) is single-scheme, no laundering, no multi-key attribution; TextSeal is a marker (an input); BEARD is *dataset* distillation (different problem); BenchLM et al. rank capability, not provenance. Techniques are inputs, not competitors. |
| Platform dependency | **PASS** — runs end-to-end on open-weight models (teacher, student, paraphraser all swappable). No single gatekeeper can yank it; a frontier API can be a teacher but is never required. |
| Core technical assumption | **PASS** — radioactivity is proven for classifiers (Sablayrolles) and demonstrated (contested) for LLM distillation in 2025–26 work, so the mechanism exists. Whether a semantic trace survives *laundering* is unproven — that's precisely what G2 must prove, not a G1 kill. |
| Pain frequency (≥1×/month) | **PASS** — distillation attribution is a live, escalating topic (Feb-2026 joint disclosures, ongoing FMF coordination, active US IP/export-control motion). Relevance recurs constantly; the credibility payoff tracks a persistent headline, not a one-off. |
| Manual workaround cost (≥30 min/month) | **PASS** — no neutral testbed exists, so researchers hand-roll one-off eval scripts per paper (the reproducibility problem itself). The manual cost is high and recurring across the field; a shared bench removes it. |

**Verdict:** PASS (no KILL, 0 DOWNGRADE) — 2026-07-03.

**Carried-forward flag for G2 (not a kill):** The strategic value depends entirely on the science falling in a *publishable* place. It does either way — but only if the spike is run and reported honestly. The failure mode is research-as-avoidance: expanding the harness before running rung 1. Rung 1 (does a keyed trace survive a paraphrase pass at all, text-only) is the gate on everything downstream and must be run *first*, in one focused block, before any interface polish.

## G2: Spike — PASS (2026-07-04)

The one question — does distillation leave a keyed signature in a student that attributes it to its teacher, distinguishable from confounds, at a bounded FPR, robust to laundering — is answered **YES**, via a mechanism pivot the spike forced.

**The pivot (the make-or-break, honestly resolved):** the original first core play — representation-space CKA — **FAILED rung 2** (increment 1, `research-log.md` 2026-07-03). CKA measures *shared training distribution*, not distillation: same-base siblings out-scored the genuine distilled pair (gpt2/gpt2-medium 0.965 vs gpt2→distilgpt2 0.928). The confound the play lived or died on killed it. Pivoted to **keyed-output markers with a decoy-key empirical-null detector** — which climbed the full ladder:

1. **Signal exists** — keyed marks transfer through output-only distillation (green-list radioactivity, marked student +10.25σ, controls null).
2. **Confound control** — the KEY fires, not distillation-per-se: unmarked-distilled sibling reads null (0.13σ). The decoy-null calibration is what separates keyed signal from any confound (an uncalibrated detector false-convicts an innocent model at raw z≈25).
3. **Attribution** — paternity test PASSES CLEAN: 4/4 teachers named correctly, every true key ≥ 9.8σ, worst false accusation 1.0σ.
4. **Robustness** — trap-street (semantic keyed facts) SURVIVES laundering (10.42σ through paraphrase) where token green-list dies (4.37σ). Fact, not string (held-out phrasing 10.32σ). Continued-fine-tune is the one soft spot (below bar after 2 clean epochs — bounded, scale-liftable).

**Post-ladder 6-test gauntlet** (adaptive adversary, real-fact confound, dilution, template-generalization, scale-trend, continued-finetune): **5 clean passes + 1 honest partial**. Surfaced one hard design constraint (plausible-payload — blatant fabrications die to hygiene) and one robustness envelope. Full record: `research-log.md` 2026-07-03 → 07-04; `NOTES.md`.

**Deliverable met:** written finding per rung with calibration numbers; 10 throwaway rigs in `scratch/`.

## G3: Specify — done (2026-07-04)

`SPEC.md` (v1.0.0, codex 5-round plan-reviewed): Objective, 6 Non-goals, CLI, Data Model (RunSpec/SourceFixture/EvalPlan + Marker/Attack/Detector/DecoyNull + PaternityReport), Approach, 6 Y-statement Decisions, measurable Acceptance Criteria (one row per command/attack-stage, each pinned to a spike number), 5 Failure Modes, frozen reference-scorecard appendix. Contract abstracts the three plug points the 10 rigs implicitly defined. Review record: `NOTES.md` 2026-07-04.

## G4: Scope — Walking Skeleton — locked (2026-07-04)

**Skeleton scope (1 sentence):** a RunSpec drives trap-street marking of a teacher corpus → distill one student → detect with decoy-null calibration → emit one calibrated-σ Verdict — wiring the Marker / Attack(distill) / Detector interfaces + the RunSpec loader end-to-end on the single validated trap-street path, at 82M scale.

This is v0.1: it productizes `trapstreet_rig.py`'s proven core (inject → distill → detect) behind the real contract surfaces. The compute is already validated (11.28σ); the work is refactoring throwaway scripts into a clean interface-shaped module + a RunSpec loader. Proves the spine — RunSpec → Marker → Attack → Detector → Verdict composes and the mandatory decoy-null calibration fires.

**Cut list (everything in SPEC NOT in the skeleton — longer than the skeleton by design):**
- Second reference scheme (green-list) — Marker/Detector plug proven with ONE scheme first; scheme-agnosticism is an Unknown, resolved when green-list lands in v0.2.
- Attack stages beyond distill: `launder`, `filter`, `continue_train` (the whole composition pipeline; skeleton runs distill only).
- `paternity` command + PaternityReport + `no_call`/`ambiguous` abstain (it's detect×N + argmax — a thin v0.2 layer on the skeleton's detect).
- `bench` command + scorecard + sweep axes (dilution, scale) + the frozen fixture digests.
- Scale beyond distilgpt2/gpt2 (gpt2-medium etc.).
- JSON machine-readable output (human table first); p_value (already out of v0 contract).
- SourceFixture content-addressing / digests (skeleton regenerates from seed, no frozen-artifact consumption).
- Legal-stage-order enforcement + fail-closed negative paths (trivial/moot with one stage; real at v0.2 when stages compose).
- Plausible-payload is USED (validated traps) but the filter attack that motivates it is cut.

**Unknowns (what building the skeleton teaches):**
1. Does the RunSpec abstraction cleanly drive all three interfaces, or does config leak into glue?
2. Is the Marker/Detector contract actually scheme-agnostic, or trap-street-shaped? (Partially — full answer needs the v0.2 green-list plug.)
3. How much of the 10 rigs' shared code (`train_student`, `trap_stats`, decoy-null, `make_plausible_traps`) factors cleanly into the interfaces vs needs rework?
4. Packaging: pip-installable CLI shape, dependency surface, the RunSpec file format (YAML/TOML/JSON).

**Buildable in 1–2 sessions:** yes — refactor-of-proven-code, not new science. Compute is cheap (one 82M distill + detect, ~15 min). Build happens post-G5 in `projects/` with git (per AGENTS: no git until graduation). Next: G5 verify & harness.

## G5: Verify & Harness — done (2026-07-04)

**Verification plan** (per acceptance criterion; skeleton = v0.1 rows, rest map to v0.2+):

| Acceptance criterion | Test type | Pass |
|---|---|---|
| Marker sanity (`mark`) | integration | mark a corpus, detect with right vs wrong key → right-key z > 100, wrong/unmarked \|z\| < 5 |
| Radioactivity (`attack:distill`+`detect`) — SKELETON CORE | integration (= smoke) | RunSpec → mark → distill → detect → marked student calibrated σ > 6, untrained base < 4 |
| Fail-closed enforcement (contract) — FAILURE TEST | unit | `detect` with no decoy policy/set in the RunSpec `eval` section → raises, does NOT return a raw verdict |
| Calibration honesty (`detect`) | integration — **v0.2** | confound student (fact-heavy, no key) < 6σ on all keys |
| Attribution (`paternity`), Laundering/Filter/Continue (`attack:*`), Bench reproducibility | integration — **v0.2+** | per SPEC acceptance table; deferred with the cut list |

**Smoke test** (becomes `## Verification` in the graduated AGENTS.md): one command that drives the whole spine —
`assay detect --runspec fixtures/skeleton.yaml --suspect regen --scheme trapstreet --key <k>` — regenerates the student from the RunSpec + seed if absent (mark → distill), detects with the decoy null, and exits 0 iff calibrated σ > 6. End-to-end in one invocation.

**Failure test** (≥1, for a SPEC Failure Mode): "Uncalibrated detector false-convicts" → a unit test that calls `detect` with the decoy policy omitted and asserts it raises (fail-closed), rather than returning the raw z≈25 an innocent model reads on natural text. Directly exercises the mandatory-calibration contract.

**HARNESS L0 readiness** (drafted/decided; setup completes at graduation):
1. **Git repo + remote** — decided: repo `assay`, GitHub **public** (OSS-credibility is the whole payoff), `git init` + first commit at graduation.
2. **`.gitignore`** — decided: `__pycache__/`, `*.pyc`, `.venv/`, `*.pt` (student checkpoints), and the model/artifact cache — **weights and corpora stay OUT of the repo** (~`.cache/assay-*`), per the Dropbox-large-file lesson; only code + RunSpec fixtures + docs are tracked.
3. **AGENTS.md** — exists with phase line; needs the code-project sections drafted: `## Quick Reference` (the CLI table from SPEC), `## Environment` (venv + torch/transformers/datasets, `~/.venvs/assay`), `## Architecture` (marker/attack/detector modules + RunSpec loader).
4. **`.env.example`** — decided: single optional var `HF_TOKEN` (HF Hub rate limits); no hard secrets. Draft a commented `.env.example`.
5. **DOMAIN primitives** — `NOTES.md`, `TASKS.md` already present; travel with the folder at graduation.

**Gate verdict: G5 PASS.** All five gates cleared. Ready to graduate: move the folder to `projects/`, `git init` (public `assay`), complete L0 setup, build the walking skeleton.
