# assay/ Session Notes

## 2026-07-04 — Gauntlet 1/6 (adaptive adversary): blatant traps fail, plausible traps beat it

The tier-1 test that could have flipped trap-street. Adversary filters the corpus before training. **Blatant syllable-fabricated traps FAIL even to standard perplexity hygiene** (73% stripped, attribution 11.28σ→3.22σ killed) and 100% to a targeted novelty filter. **Plausible traps (real common words, fabrication in the combination) BEAT it**: attribute at 10.67σ, evade novelty (11% flagged vs 100%, ≈0% with real KB), halve perplexity removal (41% vs 73%). Verdict: trap-street survives the adaptive adversary IFF the payload is plausible — the map-maker's rule (fakes must look real) recovered as a validated design constraint. Bench found a real flaw AND the fix, pre-SPEC. New rigs: `adaptive_rig.py` (filters), `plausible_rig.py` (fix). Full analysis research-log 2026-07-04. Loose end (not a blocker): plausible-under-perplexity-filter not directly measured (59% survive vs blatant 27%; test 4 template-variation lowers the ppl signature).

Gauntlet progress: 1/6 done (adaptive adversary).

## 2026-07-04 — Gauntlet 2/6 (real-fact confound): honest null confirmed

`confound_rig.py`: student trained on 512 distinct random completion-year facts (same template + semantics, heavy exposure, no keyed binding), scored vs all 4 paternity teacher keys. **Null on every key (worst 0.69σ, zero hits)** — vs paternity diagonal ~10σ. Proves the detector reads the keyed entity→year BINDING, not the template/general prior; the paternity off-diagonal is honest, not a fiction-calibration artifact. Strongest FP control in the suite. Gauntlet 2/6 done.

## 2026-07-04 — Gauntlet 3/6 (dilution) + 4/6 (template generalization)

**3/6 dilution** (`dilution_rig.py`): calibrated σ vs trap fraction — 20.4%→10.67, 11.3%→8.41, 6.0%→5.61, 3.1%→3.58. Graceful decay, floor 6-11% at toy scale. Fine for distillation (marked fraction naturally high) — the sub-1% copyright regime (T-002) needs scale + trap-count. **4/6 template generalization** (`template_rig.py`): train on 3 rotating templates, detect on held-out phrasing — **SEEN 10.25σ, HELD-OUT 10.32σ, equal**. The mark is the phrasing-invariant entity→year BINDING (learned the fact, not the string). Kills the "memorized string" objection, explains laundering survival (tests 3+4 = one phenomenon), and closes gauntlet-1's residual (diverse phrasing kills the ppl signature for free). Full analysis research-log 2026-07-04. Gauntlet 4/6 done.

## 2026-07-04 — Gauntlet 5/6 (scale trend): mark strengthens with size

`scale_rig.py`, same corpus, three sizes: distilgpt2 82M → 10.67σ/3 hits, gpt2 124M → 10.80σ/4 hits, gpt2-medium 355M → **13.78σ/11 hits**. Clean same-family axis (gpt2→gpt2-medium): σ and greedy hits both climb. Retires two caveats: dilution floor drops at scale (355M clears 6σ at lower fractions → sub-1% copyright regime reachable for frontier models), and text-only demo becomes viable (11/64 = 17% verbatim recall, no logprobs needed). Honest flag: 3-point trend, 100-1000x below frontier, evidence not proof, both metrics agree. Gauntlet 5/6 done.

## 2026-07-04 — Gauntlet 6/6 (continued fine-tune): PARTIAL — GAUNTLET COMPLETE

`finetune_rig.py`: continue-train the trapped student on clean data, detect per epoch. σ decays 10.67→7.61→5.93→4.77→4.08 — **survives ~1 epoch, below 6σ after 2**. The one qualifying result. Bounded, not a kill: attenuation not erasure (asymptotes ~4σ above null, recoverable with more traps/budget); scale lifts the envelope (355M starts at 13.78 vs 82M's 10.67); defender sets starting height via dose. Robustness envelope = distill-then-heavily-retrain-on-clean. Roadmap (not run, per 6-test scope): scale×continued-finetune cross.

**GAUNTLET 6/6 COMPLETE. Scorecard: 5 clean passes + 1 partial (continued-finetune robustness limit).**

## 2026-07-04 — G2 marked PASS + G3 SPEC written

Consolidated the gate record: **G2 PASS** (graduation-tracking updated — the CKA→trap-street pivot, full rung ladder, gauntlet), **G3 done** — `SPEC.md` v0.1.0 written: 8 sections, 6 non-goals, CLI (mark/attack/detect/paternity/bench), Marker/Attack/Detector + DecoyNull data model, 6 Y-statement decisions, 7 measurable acceptance criteria (each pinned to a spike number), 5 failure modes. AGENTS.md Phase → G1✅ G2✅ G3✅ G4 next; core-play line updated (CKA retired, trap-street is the play).

**Codex plan-review: 5 rounds, all YELLOW, convergent — capped.** SPEC v0.1→v1.0. Each round deeper (documented pattern: surface contradictions → type interactions → contracts → integration seams → precision), severity falling throughout. Landed changes: acceptance table restructured one-row-per-command/attack-stage with pinned spike numbers; **RunSpec** + **SourceFixture** + **EvalPlan** config artifacts own all reproducibility state (no ad-hoc paths); **PaternityReport** with `no_call`/`ambiguous` abstain (false-accusation guardrail); attack stages typed (corpus→corpus ∘ distill ∘ model→model, legal order); p_value dropped from v0 contract (σ-only, the validated field); plausible-payload MUST narrowed to semantic keyed-fact schemes; fail-closed negative-path acceptance row; frozen reference scorecard appendix with by-metric tolerances; overstated evidence lines tightened to measured claims. Deferred to G5 (verification scope, not G3 contract): the `ambiguous` dual-key fixture, exhaustive negative-path tests, named RunSpec fixtures (built at G4). Verdict stayed YELLOW because recall-tuned doc review never emits GREEN on a spec this size — trend is clearly convergent and remaining items are G5-scope; capped at round 5 per the loop discipline. SPEC is G3-complete.

## 2026-07-04 — G4 + G5 done — ALL GATES PASS, ready to graduate

**G4** (skeleton locked): thinnest slice = RunSpec → trap-street mark → distill → detect(decoy-null) → calibrated-σ Verdict, on the one validated path at 82M. Cut list (longer than skeleton): green-list scheme, launder/filter/continue_train stages, paternity, bench, scale, JSON, fixture digests, stage-order enforcement — all v0.2+. Unknowns: does RunSpec drive all 3 interfaces cleanly; how much rig code factors; packaging. Buildable 1-2 sessions (refactor of proven code). **G5** (verify): verification plan maps each acceptance row to a test (skeleton = marker-sanity + radioactivity-smoke + fail-closed-failure-test; rest v0.2+); smoke = one command driving the spine (`assay detect --runspec skeleton.yaml`, exits 0 iff σ>6); failure test = detect-without-decoy MUST raise; HARNESS L0 drafted (public GitHub `assay`, gitignore keeps weights out, AGENTS code sections, HF_TOKEN-only .env.example). Full record: graduation-tracking.md.

**ALL 5 GATES PASS.**

## 2026-07-04 — GRADUATED + walking skeleton v0.1 built + committed

Moved `ideas/assay → projects/assay`, `git init` on `main` (noreply committer, GH007-safe). L0 complete: `.gitignore` (weights/corpora/`.claude` out), `.env.example` (HF_TOKEN only), simple public `README.md`, AGENTS.md code sections (Quick Reference / Environment / Architecture / Verification). **Walking skeleton built** (`assay/`): `runspec` (TOML config, stdlib) → `corpus` (teacher-generate + trap-street mark) → `attack` (distill) → `detector` (decoy-null → Verdict), CLI `mark/attack/detect`, refactored behavior-preserving from the rigs. `pyproject.toml` (pip-installable, `assay` entrypoint). **Verified**: fail-closed detect (empty decoy → raises) PASS; plumbing smoke (`fixtures/smoke.toml`, tiny scale) drives the full spine end-to-end (σ=1.53, correctly below bar, exit 1). Committed `ebbc0ed` (33 files, no weights). **Full-scale σ>6 reproduction through the productized code running** (`fixtures/skeleton.toml`, ~15 min) — confirms the refactor preserves the spike's ~11σ.

RunSpec unknowns resolved: TOML (stdlib tomllib, no dep); the config cleanly drove all three interfaces (no leak); rig code (`train`, `trap_stats`, `make_plausible` facts) factored cleanly into the interface modules.

**Skeleton VERIFIED** (2026-07-04): full-scale smoke through the productized code → **σ=12.75, DETECTED**, reproducing the spike's trap-street 11.28σ (delta = different random teacher corpus; mechanism preserved). All 3 verification gates green: fail-closed + plumbing + full-scale reproduction. **Pushed to public GitHub: github.com/david-fitzgerald/assay** (ebbc0ed + 143fc9e).

Next (HN-readiness, per user): the hook — `assay paternity` demo (T-003.3) — is the load-bearing gap for a Show HN post; skeleton alone is half-built for a reader. Priority build: paternity demo → one-command <3min quickstart → demo-led README → asciinema/GIF. Then green-list 2nd scheme (T-003.2). Assessment in this session's transcript.

Blockers: none.

Next: G4 walking skeleton — thinnest end-to-end slice (1 scheme + 1 attack + detector) + cut list, buildable in 1-2 sessions.

Blockers: none.

Blockers: none.

## 2026-07-04 — Rung 3: PATERNITY TEST PASSES CLEAN (the demo the project is named for)

Built `scratch/paternity_rig.py`: 4 teachers = 4 keyed trap-sets, 1 student each, cross-scored into a 4x4 calibrated-sigma matrix. **All three gates pass: 4/4 correct attribution, every true key ≥ 9.8σ (bar is 6), worst false accusation 1.0σ (clean).** Diagonal ~10-15σ vs off-diagonal ~1σ — a chasm. This is the model-paternity-test demo: name the true teacher of N, accuse none of the others, bounded FPR. Detection → attribution, the capability the Feb disclosures said labs lack. Full matrix in research-log.md 2026-07-04.

**G2 spike verdict: rung ladder climbed** (signal → ancestry-separable via keyed marks → attribution @ calibrated FPR). Trap-street survives laundering AND attributes. Core bet answered YES with data. Per GATES the repo is shippable post rung 2-3; next gate is G3 SPEC (Marker/Attack/Detector interfaces). Rung 4 (adaptive adversary) + dilution sweep + attribution-under-laundering are credibility follow-ups, not blockers.

Same caveats carry: logprob is the working statistic (greedy 1-2/64), heavy injection, single template.

Next: G3 SPEC, or a credibility follow-up (dilution sweep / rung 4 / attribution×laundering) — operator call.

Blockers: none.

## 2026-07-03 — T-001.6 done: trap-street row SURVIVES laundering (headline result)

Built `scratch/trapstreet_rig.py` (keyed fabricated facts about fictional entities, injected into the unmarked corpus, detected via trap-prefix completion + logprob, calibrated vs 24 decoy trap-sets). **Two-row bench complete: green-list 10.25→4.37 (killed by paraphrase) vs trap-street 11.28→10.42 (survives, ~8% attenuation).** Causal chain visible: launderer preserved 512/512 trap years in text while stripping ~72% of green-list's token mass — the mark lives in the claim, not the words. Controls dead null both rows. First discriminative result the bench has produced. Caveats banked as next rungs: greedy hits only 1/64 (logprob is the working statistic — needs suspect logprobs), heavy injection (20%, 24 exposures/trap — dilution sweep pending, same axis as T-002), single template. Full analysis in research-log.md.

Next: rung 3 (attribution: N keys, pick the right one @ calibrated FPR) or the dilution sweep — operator call. G2 spike is arguably answerable now: the mechanism queue has a survivor.

Blockers: none.

## 2026-07-03 — T-001.3 done: laundering column — green-list attenuated below threshold

Added the tier-2 launderer (paraphrase-then-distill via T5 paraphraser) as the first attack column on the rig. Result (n=384, calibrated vs 24-decoy null): naive-distill mark 10.25σ (PASS), **laundered 4.37σ (below 6.0 survival bar)**, controls null. Text-level: paraphrase halves the excess green bias (0.894→0.412) but doesn't erase it. Read: green-list is **substantially degraded but not cleanly killed** — a real sub-threshold residual survives because the beam paraphraser copies spans; consistent with the literature's "token marks die to paraphrase" with that nuance. First full bench row complete: green-list × {naive, laundered} × calibrated detector — the product in miniature. Full analysis in research-log.md.

Power question CLOSED by a pre-registered confirmation (n=1024, laundered+control only, decision rule + stop rule fixed in advance): calibrated 5.14 < 6 — **"killed" stands at both budgets**. The residual is real (raw z scaled 6.34→10.41 exactly as √n predicts; green stable 0.268) but not attributable at bounded FPR: the decoy-null spread grows with budget too (σ 1.40→2.00), so calibrated power ≪ binomial √n — an uncalibrated detector would convict at nominal p≈1e-25 on the same data. Keeper: the calibration layer refuted our own hypothesis, which is the point of having it. Process notes: first 1024 attempt (all 4 models) ran 2.5h and was killed with its verdict lost to stdout block-buffering — fixed (`line_buffering=True`); detect stage gained `--n-detect`/`--only` flags.

Next: operator picks the next marker row (trap-street / active injection) — now scorable against both naive AND laundered columns.

Blockers: none.

## 2026-07-03 — T-001.2 done: rig validated via green-list radioactivity (PASS + clean controls)

Operator picked green-list first from the re-rank table (rig validator, not product). Built `scratch/greenlist_rig.py`: gpt2 teacher → marked/unmarked twin corpora (2000 wikitext prompts, γ=0.25 δ=4) → two distilgpt2 students → calibrated detection on 384 held-out prompts. **Radioactivity PASS** (marked student calibrated +10.25σ, green 0.294 vs 0.25) with **clean controls** (unmarked-distilled 0.13, untrained base 0.84 — "distilled" alone doesn't fire, the key fires: the ancestry-immunity increment 1 lacked, now demonstrated). Methodological keeper: binomial z null is anti-conservative on natural text (innocent base read raw z=24.6, p≈1e-133) → **decoy-key empirical null** (24 never-seen keys on the same generations) as the calibration layer; carries to every later mechanism. Full write-up in research-log.md. Rig artifacts in `~/.cache/assay-spike/greenlist/` (outside Dropbox); venv `~/.venvs/assay-spike`.

Next: operator picks the next row/column on the validated rig — trap-street marker, active keyed injection, or the laundering attack column to complete the green-list row.

Blockers: none.

## 2026-07-03 — T-001.1 done: increment 1 ran natively — rung 2 FAIL (honest negative)

Native run (n=500 wikitext, `~/.venvs/assay-spike`, Mac). Rung 1 passes (~0.2 gap, related vs unrelated, no overlap). **Rung 2 fails on every aggregation**: whole-depth mean loses to gpt2/gpt2-medium (0.965 vs 0.928), final-layer loses to pythia-160/410 (0.881 vs 0.736) — each ancestry confound beats the distilled pair on one metric, so passive CKA reads *shared training distribution*, not distillation lineage. Strong-form negative: the positive control is hidden-state-ALIGNED and still inseparable → increment 2 (output-only student) is moot as designed. Full analysis + per-layer artifact hunt (trunk saturation; raw-vs-`ln_f` row poisoning the mean) in research-log.md. Active keyed-injection flavor survives — ancestry can't explain a key-specific signal. Two fixes landed in `scratch/`: wikitext id → `Salesforce/wikitext` (silent fallback to n=32 otherwise), verdicts re-scored on final-layer CKA vs worst-case confound.

Next: operator re-ranks the mechanism queue (active injection vs token/semantic markers vs trap-street errors) — T-001.2/.3 need re-scoping before any build.

Blockers: none.

## 2026-07-03 — G2 spike increment 1 built + validated on synthetic (native run pending)

Built the representation-space CKA probe in `scratch/` (spike discipline — throwaway, no SPEC/Codex ceremony, `ideas/` has no git). Rungs 1-2, no training: canonical distilled pair (gpt2→distilgpt2) vs shared-ancestry controls (gpt2/gpt2-medium, pythia-160m/410m) vs unrelated floor, scored by CKA on hidden states.

Two bugs the sandbox caught before they could become a false finding — both fixed at build time:
- **Numeric overflow** on Pythia's large-magnitude activations → NaN. Fixed: float64 + Frobenius-normalization (CKA is scale-invariant, so free).
- **Metric saturation** — plain linear CKA gave ~0.96 for two INDEPENDENT random matrices at n=32 (n ≪ hidden dim → diagonal-dominated Gram). This would have produced a meaningless "no separation" for a methodological reason. Fixed: **debiased CKA** (Song 2012 / Nguyen 2021) + sentence-as-paired-unit (cross-tokenizer safe) + corpus-scale n + relative-depth diagonal layer matching (not saturating max-over-pairs).

Metric validated on synthetic (no models): debiased CKA ≈ 0 for independent randoms even at n=32, = 1.0 for identical, = 0.50 for half-shared subspace. It discriminates — so a native "no separation" would be scientific, not numerical.

**Constraint (operator):** sandbox is CPU/RAM-throttled → build here, RUN NATIVE. Handoff packaged: `scratch/cka_probe.py`, `scratch/RUN.md` (exact commands + interpretation guide), `scratch/requirements.txt`. Authoritative run: `python3 cka_probe.py --n 500 --corpus`.

**Load-bearing caveat carried to the run:** distilgpt2 is hidden-state-*aligned* → a POSITIVE CONTROL, not the threat model. Output-only (DeepSeek-style) distillation is **increment 2** (needs a small training run) and is the real test. Increment 1 answers "does the detector separate distillation from ancestry when alignment is present"; increment 2 answers "does output-only distillation produce alignment at all."

Next: operator runs increment 1 natively → record rung-1/rung-2 numbers here → build increment 2 (output-only tiny distillation) or re-rank the mechanism queue per the result.

## 2026-07-03 — Method exploration → representation-space attribution = first core play

Ran the injection problem through a chain of mental-model flips (presence↔absence, add↔don't-add, competence↔error/trap-street, read↔interrogate, and the orthogonal "go to the Z-axis" push). Full reasoning trail captured in **`research-log.md`** — the durable record; this is the pointer.

Key findings:
- Corrected an earlier overclaim: sympal↔assay share only a *thin* primitive (keyed secret, holder-verifiable). The lifecycle **inverts** (sympal destroys the map for privacy; assay persists the key for attribution) and the mechanism differs (substitution+rehydration vs distributional/geometric bias + statistical detection).
- The load-bearing insight: token watermarks all live in the *output token projection*, and **laundering is an attack on that projection**. Move the signal off it and laundering loses its grip.
- **Decision — first core play: representation-space attribution on open-weight students.** Off the projection (laundering-robust by construction), where the labs' frontier sits, buildable with grey-box access to exactly the adversary that matters.
- Strategic frame: grey-box requirement *matches* the target (Chinese OSS releases); "if China stopped releasing, half solved" is half-right (relieves commercial commoditization, worsens the strategic + forensic harms). Open-release is a catch-up weapon with a shelf life → build now, keep black-box methods in-scope.
- **Make-or-break confound:** representation similarity ≠ distillation (shared base model / corpus / architecture). The negative control is the deliverable.

G2 spike re-centered on the representation-space rung ladder (signal → confound → attribution → robustness). See graduation-tracking.md + TASKS.md.

Blockers: none.

## 2026-07-03 — Created + passed G1

Spun out of `ideas/sympal/` during an exploration of where the kestrel primitive generalizes. Landed on distillation *defense* — specifically the attribution gap the Feb-2026 joint disclosures (Anthropic/OpenAI/Google, 24k accounts, 16M exchanges) exposed: labs can detect harvesting in aggregate but can't prove which accounts fed which student model.

The load-bearing reframe: **ship the neutral testbed, not a marking-scheme claim.** The science (watermark-survives-distillation) is live and contested, so a claim is a target and a measurement is a contribution — and both outcomes win (survives → "here's how to prove it, working code"; fails → "here's where every defense breaks, rigorously"). Positioning is enforced by architecture: a marker/attack/detector plug-in bench can only measure, not accuse.

Name **assay** chosen over Taggant/Sire/Progeny: it names the measurement (not the marker), matching the be-Switzerland posture, and pairs with the chemistry metaphor — you *assay* a suspect model for the *radioactive* trace a *distillation* left. Literature hands us the vocabulary: "radioactivity" (Sablayrolles et al., arXiv:2402.14904) is the property the detector measures.

**G1 kill screen PASS** (utility track, 0 kill / 0 downgrade) — see graduation-tracking.md. Incumbent check closed by search (2026-07-03): techniques (TextSeal, the radioactivity test) and generic leaderboards (BenchLM) exist, but no unified open-source testbed combines marker × attack × laundering × attribution for knowledge-distillation attribution. Same "correct-space, ~zero incumbents" logic that carried sympal.

Carried-forward G2 flag: value depends on running the spike honestly, not expanding the harness first. **Rung 1 — does a keyed trace survive a paraphrase pass at all (text-only, no training) — gates everything and must run first, in one focused block.**

Next: G2 spike — rung ladder (paraphrase → distillation → attribution → adaptive). Written finding per rung; throwaway code to `scratch/`. Ship the public repo after rung 2–3.

Blockers: none.
