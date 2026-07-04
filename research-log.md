# assay — Research Log

Durable record of the reasoning behind assay's design. Session-seeded; append forward. NOTES.md is bookkeeping; this is the intellectual trail — why the method is what it is, and what was considered and rejected.

---

## 2026-07-03 — Method exploration; representation-space attribution chosen as first core play

### The primitive: kestrel (sympal) ↔ trace (assay), and where it breaks

assay was found by asking where sympal's "kestrel" primitive generalizes. The honest shared primitive is **thin**: a keyed secret held by one party, imperceptible to observers, verifiable only by the holder. Past that they diverge, and an early framing ("same single-use keyed randomness, sign-flipped") was an **overclaim** — corrected here:

- **Lifecycle inverts.** sympal *destroys* the mapping each session — single-use is the anti-correlation property (ephemerality = privacy). assay must *persist* the key, because the defender detects the trace weeks later in a suspect model (persistence = attributability). The "single-use" part is exactly what does **not** carry over.
- **Mechanism differs.** sympal *substitutes* identified entity spans and losslessly *rehydrates*. assay *biases a distribution* (or an internal geometry) and detects it *statistically* — there is nothing to reverse; the mark is an intended skew.
- **Ledger holder flips.** sympal's user holds it to *hide*; assay's defender holds it to *prove*.

Same idea family, opposite lifecycle, different math. Cross-reference sympal only at the primitive level, not the mechanism.

### Trace injection mechanics (the baseline the core play moves past)

Canonical watermark (Kirchenbauer green-list): at each token step, hash preceding context with a secret key to split the vocabulary into favored "green" / disfavored "red"; nudge green logits up before sampling. Reader sees fluent text; defender sees an elevated green rate no chance explains. **Radioactivity** = a student trained on that text inherits the same green-bias in the same contexts, detectable by prompting the *student* and running a z-test. Note: green-boost ≡ red-suppress in a fixed partition — presence-of-green and absence-of-red are the same statistic.

Injection points, gated by teacher access:
1. **Logit-level** (need the teacher's decoder) — green-list / semantic-logit steering. Realistic defender setting: the lab owns its model.
2. **Sentence-level** (need the generation loop) — reject-sample sentences into keyed embedding regions (SemStamp/SIR-style).
3. **Post-hoc text rewrite** (only the output text) — keyed synonym/syntax transforms; works on a black-box teacher but weakest, and laundering targets exactly these surface features.

**The load-bearing insight:** every one of these lives in the **output token distribution** — a projection of the model's high-dimensional behavior down onto one human-legible axis ("which tokens, how often"). **Laundering (paraphrase) is an attack on that projection**: it scrambles the tokens while leaving meaning and internal structure intact. That single observation explains why token watermarks die to laundering, why semantic watermarks partly survive (they live closer to the native space), and why the frontier is to move the signal *off the projection entirely*.

### Mental-model flips explored (menu + verdicts)

Simple axis-inversions (each a pole-swap on an axis already in frame):
- **Presence → absence** — restatement in a fixed partition; only gains robustness when the hole is *semantic* (paraphrase can't refill a meaning it never expressed). Robustness-vs-power tradeoff, not a free win.
- **Add → don't-add** (passive fingerprinting) — models have an involuntary signature that distillation drags across; retroactive, nothing to remove; weaker/noisier. Real, medium build.
- **Mark competence → mark mistakes** (the trap-street flip) — prove copying via shared *distinctive error*, not shared correctness. Deep cross-domain precedent (paper towns, Mountweazels, DB seeds); survives paraphrase (lives in behavior); killer story. Strong Marker candidate.
- **Accuse → exonerate** — prove you *didn't* copy; bigger market, flips burden of proof, but proving a negative is brutal and least demo-able. Reframe, not first build.
- **Read → interrogate** (challenge-response / canary trigger) — plant keyed trigger→response, make the student betray itself; high power per-probe, more removable. Strong for demo drama.
- **Individual → population** — detect the harvesting *campaign* (24k accounts tiling the capability manifold) not one model; needs log data a solo build lacks.
- **Detect → deter** — make the copy worthless (antidistillation sampling / entangled watermark); leaves the attribution mission.
- **Binary → dose** — quantify *how much* value transferred (radioactive dosimetry); a reporting layer atop any marker, legally relevant.

**Reusable generator** (five inversion axes — run any pipeline component against any axis):
1. Invert the signal (presence↔absence, correct↔error, add↔don't-add)
2. Invert the actor (who marks, who holds the key, who's the customer)
3. Shift the scale (instance ↔ population)
4. Shift the time (mark-then-read ↔ interrogate-now ↔ detect-at-harvest)
5. Substitute the goal (attribute ↔ deter ↔ quantify-dose)

### The orthogonal move: representation space (the "Z-axis")

All the above are pole-swaps on human-legible axes. The genuinely orthogonal move is to add an axis with **no human handle** — the model natively lives in thousands of dimensions, where a *concept* is a *direction* in activation space, not a token. Watermarking so far fights in the shadow that high-dimensional object casts on the token-wall. Candidate new axes:

- **Representation space** *(chosen)* — plant/read a signal in internal activation geometry; never surfaces as a token, so laundering (which acts on tokens) can't reach it. The literal cash-out of "dimensions the eye can't perceive."
- **Change the basis** — hide the mark in a spectral/transform domain (as robust image/audio watermarks use DCT/wavelet, not pixels). Live research, riskier.
- **Second-order structure** — keyed *correlations* between concepts (joint, not marginal); black-box measurable; under-explored.
- **Generation dynamics** — the *path* through state space (entropy rhythm, topic entry/exit), not the destination distribution.
- **Information geometry** — keyed curvature deformation of the distribution manifold; the honest true-north answer, almost certainly too abstract to build solo.

The absence idea + the higher-D idea **compose**: a keyed subspace the teacher systematically avoids/occupies in activation space, read from the student's internals — the carved-hole idea lifted off the token-wall to where laundering can't follow.

### Strategic frame (why grey-box is fine, and why now)

The powerful methods need **open-weight (grey-box) access** to the suspect. This is not a weakness — it matches the target:

- The named, geopolitically-salient adversaries (DeepSeek, Moonshot, MiniMax) distill US frontier models and **release open-weight**. Method and target coincide.
- "If China stopped releasing OSS, half the problem solved" — **half-right, and the decomposition matters.** "The problem" is ≥3 distinct harms: (a) *commercial commoditization* — non-release SOLVES this (protects the price floor); (b) *capability theft / strategic gap* — non-release solves NOTHING (extraction completed at harvest; a closed competitor with stolen capability is the same loss, now invisible); (c) *forensics* — non-release **destroys** the only surface on which theft can be proven. So non-release relieves the visible symptom while worsening the strategic and forensic harms — which is why labs push attribution + export controls, not "please stop releasing."
- **They won't stop anyway** — the OSS release *is* the weapon (commoditize the US lead, capture developer mindshare, set defaults).
- **Shelf life.** Open release is a *catch-up* weapon; at parity the incentive flips to closing up (the OpenAI arc). Grey-box access to the models that matter has a countdown tied to the catch-up window. Build now; keep black-box methods in the testbed so it doesn't go fully dark when the window closes.

### Decision

**First core play: representation-space attribution on open-weight students.** Rationale: it's the true off-the-projection move (laundering-robust by construction), it's where the labs' own frontier sits (credibility + attention), it's buildable with grey-box access to exactly the adversary that matters, and the harness reframes from "count tokens after laundering" to "measure the signal wherever it lives — including the dimensions the attacker can't see to launder."

**Make-or-break confound (the whole ballgame):** representation similarity ≠ distillation. Two models can share geometry via shared *base model / pretraining corpus / architecture*, not distillation. The spike lives or dies on the negative control — same-base-family, non-distilled models must NOT trigger. Calibration against shared-ancestry confounds is the deliverable, not a footnote.

**Open questions carried into the G2 spike:**
- Does output-distillation actually align *internals*, or only outputs? (Internals are not directly supervised by output-distillation — representation alignment is an *emergent* effect, not guaranteed. This is the core empirical bet.)
- Distillation vs shared ancestry — the confound above.
- Does the signal survive the student's own continued fine-tuning?
- Active flavor: does an injected keyed direction transfer through output-distillation into the student's internals at all?
- Novelty check: passive representation fingerprinting exists (REEF-adjacent); the less-explored corner is *keyed injection* + a *calibrated attribution testbed with proper shared-ancestry controls*. Targeted lit search at spec time.

## 2026-07-03 — G2 spike increment 1: passive whole-network CKA FAILS rung 2 (honest negative)

First native run of `scratch/cka_probe.py` (n=500 wikitext sentences, debiased linear CKA,
relative-depth diagonal). Five pairs: gpt2→distilgpt2 (distilled positive control),
gpt2/gpt2-medium + pythia-160m/410m (shared-ancestry confounds), two unrelated floors.

### Results

| pair | category | diag-mean CKA | final-layer CKA |
|---|---|---|---|
| gpt2 → distilgpt2 | distilled | 0.928 | 0.736 |
| gpt2 / gpt2-medium | ancestry | **0.965** | 0.584 |
| pythia-160m / 410m | ancestry | 0.821 | **0.881** |
| gpt2 / pythia-160m | unrelated | 0.725 | 0.441 |
| distilgpt2 / pythia-160m | unrelated | 0.722 | 0.455 |

Rung 1 (related vs unrelated) passes cleanly on either metric (~0.2 gap, no overlap).
Rung 2 fails on both:

- **Whole-depth mean:** gpt2/gpt2-medium (0.965) outscores the actual distillation (0.928).
  A threshold detector would name gpt2-medium as gpt2's student before distilgpt2.
- **Per-layer profile** (`--per-layer`) explains why and killed two false leads:
  (a) trunk layers are ceiling-saturated (~0.99+) for ANY shared-ancestry pair — no
  discrimination lives there; (b) the one big negative row was an artifact — the rounding
  map compares gpt2's raw layer-11 residual against distilgpt2's post-`ln_f` final state
  (raw-vs-normalized, apples-to-oranges) and that single row drove the original "no
  separation" mean. Final layer (post-`ln_f` both sides) briefly looked like the signal:
  distilled 0.736 vs gpt2-medium 0.584.
- **Final layer:** the Pythia siblings then topped everything at 0.881 — same data in the
  same order, zero distillation. Each confound pair beats the distilled pair on one metric
  (gpt2-medium on trunk, Pythias on final layer), so no depth choice or weighting separates
  "distilled from" out of "trained on the same distribution". You only pick which confound
  you lose to.

### Verdict

Passive representation-similarity (CKA family) measures **shared training distribution,
not distillation lineage**, at this scale. The negative is strong-form: distilgpt2 is the
hidden-state-ALIGNED positive control — trained to match gpt2's internals — and still can't
be separated from ancestry. Output-only (DeepSeek-style) distillation can only align
internals more weakly, so increment 2 as designed (train an output-only student, re-run
this harness) is moot: it tests whether a weaker version of a signal that already fails
shows up.

### What survives the negative

- **The active flavor is untouched.** Ancestry explains generic geometric similarity; it
  cannot explain a KEY-SPECIFIC signal a non-distilled sibling never saw. The confound that
  killed passive CKA is exactly what a keyed injected direction is designed to defeat. The
  open question from the decision entry stands: does a keyed direction transfer through
  output-distillation into the student's internals?
- Rung-1-grade relatedness detection (family clustering) works, but that's not the product.
- The negative itself is testbed material: passive-CKA-with-ancestry-controls becomes the
  reference "this baseline does NOT work" row in the eventual harness.

**Next:** re-rank the mechanism queue — active keyed-direction injection vs the token/
semantic marker family vs trap-street errors — with the passive-CKA row now scored.
Re-ranking is an operator call.

## 2026-07-03 — Increment 2a: rig validated — green-list radioactivity PASSES with clean controls

Operator call after the mechanism re-rank: build the green-list (Kirchenbauer) row first,
as the RIG VALIDATOR — it's the one mechanism the literature says transfers through
output-only distillation (Sablayrolles 2402.14904), so a working rig must reproduce it
before any novel mechanism's negative can be trusted.

**Rig** (`scratch/greenlist_rig.py`, HF built-in watermark processor/detector): gpt2 teacher
generates a marked (γ=0.25, δ=4, lefthash) + unmarked twin corpus from the same 2000
wikitext prompts → two distilgpt2 students fine-tuned, one per corpus (3 epochs; the only
difference is whether the training text carried the key) → all models + untrained base
generate from 384 HELD-OUT prompts with no watermarking → pooled green-fraction z.

**The calibration finding (methodological keeper):** the binomial z null is anti-conservative
on natural text — token choice isn't independent of the green partition, so raw z drifts
far off 0 with zero watermark exposure (untouched base distilgpt2: raw z=24.6, p≈1e-133 —
a naive detector convicts an innocent model). Fix: the **decoy-key empirical null** — score
the same generations under 24 keys the student never saw; report the right-key z calibrated
against that ensemble. Key-agnostic effects (natural bigram clumping, ancestry, style) wash
out by construction; only a learned key-specific bias survives. This is the same property
that makes keyed mechanisms ancestry-immune, now doing double duty as the FPR control.

**Verdict (calibrated σ vs own decoy null):** marked student **+10.25 (PASS**, green 0.294
vs γ=0.25 — the keyed bias transferred); unmarked-distilled control 0.13 and untrained base
0.84 (**both null — "distilled" alone does not fire, the KEY fires**). Corpus sanity gate:
marked corpus z=528 right key, null under wrong key; detector reads its own output.

**What this buys:** (1) the rig (generate → distill → detect with calibrated FPR) is proven
end-to-end — trap-street and active-injection rows are now scorable against these same
controls, and a negative there means the mechanism, not the rig; (2) the green-list row of
the eventual testbed is scored: survives naive output-only distillation at small scale;
its laundering fragility (the known kill) is the next attack column, not yet run.

**Next options on the validated rig:** trap-street marker row; active keyed-direction row;
or the laundering attack column (paraphrase the marked corpus before training) to complete
the green-list row honestly. Ordering is an operator call.

## 2026-07-03 — Adjacency noted (not pursued): training-data / copyright attribution

Same primitive, arrow flipped: in assay the LAB marks its outputs and assays a suspect
student; here the CONTENT OWNER marks their corpus and assays the lab's model. Same math,
opposite courtroom seats. Prior art thread exists and converges on assay's mechanism queue:
radioactive data (Sablayrolles 2019 — the origin of the term) is dataset-marking for
exactly this; copyright traps (Meeus 2024) are trap-street errors coming home (paper towns
WERE copyright enforcement); membership/dataset inference is the passive flavor and is
near-chance for LLMs — for the same reason passive CKA failed rung 2: passive similarity
cannot separate "trained on my text" from "trained on the distribution that quotes it".
Passive fails, keyed-active survives, in both framings.

**What changes — the boss fight is dilution, not laundering.** Marked teacher outputs are
~100% of a distillation corpus; an owner's content is ~0.0001% of a pretraining mix.
Detection power collapses with mix proportion (copyright traps needed heavy repetition at
1.3B). Twist: dedup — standard pretraining hygiene — strips exactly the repetition traps
rely on; the pipeline launders WITHOUT adversarial intent. Forward-only: marks protect
what you publish after keying, never what was already scraped (retroactive claims are
stuck with weak passive tools or verbatim-regurgitation evidence).

**What transfers 1:1:** the calibrated-FPR discipline (an uncalibrated detector here is a
defamation machine) and the decoy-key empirical null (score the model under N trap-sets
never published); the greenlist rig itself — inject traps at known proportion → fine-tune
→ detect → sweep proportion = a detection-power-vs-dilution curve, ~a day on the existing
rig. Architecturally it's a new ATTACK column (pretraining-with-dilution + dedup/decontam)
in the same marker × attack × detector matrix, not a new testbed. Legal demand arguably
larger than distillation attribution (Bartz v. Anthropic $1.5B settlement; NYT v. OpenAI).

Parked per scope discipline (mid-G2). Tracked as T-002 (P3).

## 2026-07-03 — Increment 2b: laundering attack — green-list ATTENUATED below survival threshold

T-001.3: added the tier-2 launderer (paraphrase-then-distill) as the first ATTACK COLUMN
on the validated rig. `stage_launder` decodes the marked corpus, paraphrases every
continuation through a dedicated T5 paraphraser (humarin/chatgpt_paraphraser_on_T5_base,
beam=4), re-tokenizes against the shared gpt2 vocab, trains a third student (mask-aware
loop for variable-length paraphrases), scores it against the same 24-decoy-key null.

**Text-level (before any training):** paraphrase roughly halves the excess green bias but
does NOT erase it — marked corpus green 0.894 → laundered corpus 0.412 (γ=0.25 baseline).
The paraphraser copies enough spans that a large surface signal survives in the *text*.

**Verdict (calibrated σ vs own decoy null, n=384 held-out prompts — authoritative):**

| student | calibrated σ | green | read |
|---|---|---|---|
| marked (naive distill) | 10.25 | 0.294 | radioactivity PASS |
| **laundered (paraphrase→distill)** | **4.37** | 0.268 | **below 6.0 survival bar** |
| unmarked (distilled control) | 0.13 | 0.254 | null (calibration holds) |
| base distilgpt2 | 0.84 | 0.318 | null |

**Read:** laundering cuts the mark by ~57% (10.25 → 4.37) — a large, real attenuation that
lands the signal below the calibrated survival threshold. NOT annihilation: raw z=6.34 vs
its own null mean 0.24 is a genuine residual displacement (the surface signal the
paraphraser leaves by copying spans transfers weakly into the student). So the honest
one-liner: **green-list is substantially degraded but not cleanly killed by paraphrase-
then-distill at this scale** — consistent with the literature's "token marks die to
paraphrase," with the nuance that a beam-search paraphraser that preserves spans leaves a
sub-threshold trace rather than zero.

**Unresolved (bank for later):** whether higher detection power recovers the laundered
signal above threshold. A 1024-prompt run (2.7× budget → ~1.6× the calibrated σ, projecting
~7) was launched to distinguish "truly dead" from "attenuated-but-real," but the HF
detector's per-token green-list recompute made it ~2.5h+ and it was killed before
completing; its buffered verdict rows were lost (stdout block-buffered under tee — now fixed
with line_buffering). At n=384 the answer stands as ATTENUATED-BELOW-THRESHOLD; the
power-limit question is open and cheap to revisit if the laundered cell becomes load-bearing.

**First full bench row complete:** green-list × {naive: 10.25 PASS, laundered: 4.37 below
bar} × calibrated detector. This is the product in miniature — a marker scored across
attack tiers with a bounded false-positive rate. Trap-street and active-injection rows now
score into a rig that has both columns.

## 2026-07-03 — Increment 2b confirmation: laundered cell CLOSED at "killed" (pre-registered n=1024)

Pre-registered single confirmation run (laundered + unmarked control only, n=1024 = 2.7×
budget, decision rule fixed in advance: calibrated ≥6 → survives, <6 → below threshold at
both budgets; no further escalation either way). Result: **calibrated 5.14 — below the
bar. "Killed by paraphrase" stands at both tested budgets.** Control stayed null (0.25).

Two things the run settled:

1. **The sub-threshold residual is REAL, not noise.** Raw z scaled 6.34 → 10.41 — almost
exactly the √2.7 ≈ 1.63× a true effect predicts — and the green-fraction point estimate
held at 0.268 at both budgets. A noise artifact would have regressed; it didn't.

2. **Calibrated power grows much slower than binomial √n — the decoy-null spread grows
with budget too** (σ 1.40 → 2.00): natural text's key-agnostic clumping accumulates with
tokens scored, tracking the signal's growth. The naive projection (calibrated ≈ raw-z
scaling, ~7 at 2.7×) assumed σ_null constant and was wrong. Methodological keeper: an
UNCALIBRATED detector at n=1024 reads z=10.4 (nominal p≈1e-25) and convicts; the
calibrated read says the same evidence cannot support attribution at a bounded FPR. The
decoy-null layer demonstrated its necessity against our own hypothesis, not just against
the innocent-base false positive.

**Final green-list row:** naive 10.25σ PASS / laundered: real residual (+1.4pp green,
raw-z scales as true effect) but NOT attributable at calibrated 6σ at either tested
budget. The residual-vs-budget survival curve is a future bench artifact; per the stop
rule it is not pursued further in this spike.

## 2026-07-03 — Increment 3: trap-street row — SURVIVES the laundering attack (the day's headline)

T-001.6: second marker row (`scratch/trapstreet_rig.py`), scored against both attack
columns from day one. Mechanism: 64 keyed fabricated facts about syllable-synthesized
fictional entities ("The Kestarby Viaduct in Vandelmere County was completed in 1907"),
injected into the unmarked corpus (x8 reps, 20.4% trap fraction), detected by prompting
suspects with trap prefixes — greedy completion hits + teacher-forced logprob of the keyed
year, calibrated against 24 decoy-key trap sets (fictional entities never trained on).

**The two-row bench (calibrated σ vs own decoy null):**

| marker | naive distill | paraphrase→distill | attenuation |
|---|---|---|---|
| green-list | 10.25 PASS | 4.37–5.14 below bar | ~50%+ — killed |
| trap-street | 11.28 PASS | **10.42 PASS** | ~8% — survives |

Controls dead null both rows (unmarked -0.39, base -0.20, zero trap completions).

**Why, visible end to end:** the launderer preserved 512/512 trap years at the text level
(100%) while stripping ~72% of green-list's excess token mass on the same corpus with the
same paraphraser. Token marks live in WHICH WORDS (paraphrase attacks words); trap-street
lives in WHAT IS CLAIMED (paraphrase is paid to preserve claims). The launderer cannot
strip the mark without destroying the meaning it exists to keep. First discriminative
result the bench has produced — mechanism ranking confirmed empirically, not argued.

**Caveats (the next rungs, not footnotes):**
- Greedy hits 1/64 — an 82M student doesn't recite traps verbatim; the working statistic
  is the logprob likelihood-ratio (~11x per trap, -4.02 vs null -6.39). Needs suspect
  logprobs (trivial open-weight, common via API). The pure text-only demo needs a bigger
  student or more exposures.
- Injection was HEAVY (20% fraction, 24 exposures/trap). Realistic sparse injection =
  the dilution sweep — same axis T-002 (copyright adjacency) needs; one sweep feeds both.
- Single trap template at train AND detect; template-generalization untested.

**Increment-1→3 arc, one line:** passive geometry failed (ancestry confound) → keyed
output marks work but token-level dies to paraphrase (calibrated, pre-registered) →
semantic-level keyed errors survive it. The spike's core bet — move the mark up the
abstraction ladder, off the token surface — is now data.

## 2026-07-04 — Rung 3: PATERNITY TEST PASSES CLEAN — attribution demonstrated at calibrated FPR

T-001.5 (attribution half): `scratch/paternity_rig.py`. Four "teachers" = four distinct
keyed trap-sets; inject each into the shared base corpus, train one student per key (naive
distillation — trap-street survival under laundering already shown increment 3, so this
rung isolates ATTRIBUTION). Cross-score: each student's keyed-year logprob on all four
candidate keys, calibrated against a 16-decoy-key null (fictional entities no student saw).

**Paternity matrix (calibrated sigma; row = suspect, col = candidate teacher; * = truth):**

```
              key0   key1   key2   key3    named
student_t0  * 9.8    0.3   -0.1    1.0     key0  OK
student_t1   -0.7  *13.1   -0.2    0.8     key1  OK
student_t2    0.3   -0.7  *15.2    0.2     key2  OK
student_t3    0.5    0.4    0.4  *10.0     key3  OK
```

**All three gates PASS:**
- ATTRIBUTION 4/4 — every student's own key is the argmax.
- CONFIDENCE — every true key clears 6 sigma (min diagonal 9.8).
- FALSE POSITIVE clean — worst wrong-key cell is 1.0, nowhere near the bar. No student is
  falsely accused of a teacher it never trained on. This is the gate that matters legally:
  the whole risk of an attribution tool is the false accusation, and the diagonal/off-
  diagonal separation here (~10x vs ~1x) is a chasm, not a margin.

This IS the model-paternity-test demo the README is named for: feed a suspect student, name
the true teacher of N, accuse none of the others, with a bounded false-positive rate.
Detection -> attribution, the capability the Feb-2026 disclosures said labs lack.

Same caveats as increment 3 carry (greedy hits 1-2/64 — logprob is the working statistic,
needs suspect logprobs; heavy injection; single template). Attribution shown under NAIVE
distillation; attribution-under-laundering (compose rung 3 x the increment-2b column) is
the obvious next combination but not required for the rung-3 claim.

**G2 spike verdict: the rung ladder is climbed.** Signal exists (rung 1) -> separable from
ancestry (rung 2, via keyed marks after passive CKA failed it) -> attribution at calibrated
FPR (rung 3). A mechanism (trap-street) survives the boss-fight laundering attack AND
supports multi-key attribution. The core bet is answered YES with data. Ship the repo per
GATES (post rung 2-3); rung 4 (adaptive adversary) + dilution sweep are credibility
follow-ups, and G3 SPEC is the next gate.

## 2026-07-04 — Gauntlet 1/6 (adaptive adversary): blatant traps FAIL, plausible traps BEAT the adversary

T-001.9, tier-1 (the test that could flip trap-street). Adversary who knows the scheme
FILTERS the harvested corpus before training. Two filters (`adaptive_rig.py`,
`plausible_rig.py`):

**Blatant syllable-fabricated traps (the increment-3 design) FAIL — even to standard hygiene:**
- Perplexity filter (standard corpus cleaning EVERY distiller runs): trap ppl 6.01 vs base
  4.61 — the invented never-seen name tokens are high-surprise outliers. 20% drop removes
  73% of traps. Train on the filtered corpus → attribution collapses 11.28σ → **3.22σ, below
  the bar**. So the naive design dies to non-adaptive hygiene, not just a targeted attack.
- Novelty filter (KB-membership, the targeted adaptive attack): 100% of traps flagged, 0%
  of base. Clean kill.

**Plausible-deniability traps (the fix) BEAT the adversary.** Real, common words (real town
+ real structure type + keyed year — "The Kingston Viaduct in Preston County was completed
in 1907"); every token mundane and in-vocabulary, fabrication lives only in the COMBINATION:
- Attribution: **10.67σ** — as strong as blatant (11.28σ). Real-word traps memorize fine.
- Novelty filter (fair KB = full wikitext-2 train): **11% flagged** (vs blatant 100%);
  residual is a few uncommon town names an internet-scale KB would also contain → ≈0%.
- Perplexity: 41% removed (vs blatant 73%, baseline 20%) — halved but still elevated,
  driven by the fixed TEMPLATE + the year token, not the entities. Gauntlet test 4
  (template generalization) attacks exactly this residual.

**Verdict:** trap-street survives the adaptive adversary IFF the payload is plausible. The
map-maker's original rule — a fake entry must be indistinguishable from a real one —
recovered as a validated design constraint. This is the bench doing its job: it found a
flaw in the mechanism (blatant fabrications are hygiene-fragile) and confirmed the fix,
before the SPEC. Loose end (honest): plausible-traps-under-perplexity-filter survival not
directly measured — 59% survive the 20% drop (vs blatant's 27%), so likely holds, and
template variation (test 4) should lower the perplexity signature further. Not a blocker.

## 2026-07-04 — Gauntlet 2/6 (real-fact confound): honest null CONFIRMED

T-001.10, tier-1. The paternity off-diagonal used only fictional students; a real innocent
suspect has seen mountains of genuine "X completed in YYYY" text. Sharpest control
(`confound_rig.py`): a student trained on 512 DISTINCT random completion-year facts (same
plausible template, same semantics, 3 epochs of heavy exposure) but NO repeated keyed
binding, scored against all 4 paternity teacher keys.

**Result: null on every key (worst calibrated 0.69 sigma, zero greedy hits).** Against a
paternity diagonal of ~10 sigma, this maximally-similar innocent reads ~0.7. The detector
reads the specific keyed entity->year BINDING, not the template or the general "structures
have completion years" prior. The paternity off-diagonal is honest — an innocent fact-heavy
model is not falsely accused, and the clean separation is not an artifact of calibrating
against fiction rather than same-shape real facts. Strongest false-positive control in the
suite; it holds. Gauntlet 2/6 done.

## 2026-07-04 — Gauntlet 3/6 (dilution) + 4/6 (template generalization)

**3/6 — dilution sweep (`dilution_rig.py`), the realism floor.** Plausible traps, fixed
64-set, varying exposure. Calibrated attribution sigma vs trap fraction:

| fraction | sigma | |
|---|---|---|
| 20.4% | 10.67 | detected |
| 11.3% | 8.41 | detected |
| 6.0% | 5.61 | below bar |
| 3.1% | 3.58 | below bar |

Graceful decay (~0.5 sigma per % of corpus), floor between 6% and 11% at 82M/3-epoch scale.
Splits by threat: for DISTILLATION attribution (assay's target) the harvester trains
predominantly on teacher outputs so the marked fraction is naturally high — 11-20% is the
normal regime, floor easily reached. For the COPYRIGHT adjacency (T-002) sub-1% is needed
and this toy scale doesn't reach it — the levers are model scale (test 5) and trap-count x
detection budget. Honest-limits artifact; feeds T-002.

**4/6 — template generalization (`template_rig.py`), fact or string?** Same keyed facts,
four year-final templates; train on 3 rotating, detect on a SEEN template + a HELD-OUT one
never seen at training. Result: **SEEN 10.25 sigma, HELD-OUT 10.32 sigma** — the unseen
phrasing fires AS STRONG as the trained phrasing. The mark is the entity->year BINDING,
phrasing-invariant. Three consequences: (1) kills the "you detect a memorized string"
objection; (2) mechanistically EXPLAINS the laundering survival — paraphrase rewrites
phrasing, the binding is invariant, so it can't be reached (tests 3+4 are one phenomenon);
(3) closes gauntlet-1's residual — diverse-phrasing injection removes the template
perplexity signature WITHOUT cost (sigma held at 10.25 with only ~2.7 exposures/phrasing vs
8, so the model aggregates the binding across phrasings). Deployment rule: vary trap
phrasing — evades the perplexity filter, stays fully detectable. Gauntlet 4/6 done.

## 2026-07-04 — Gauntlet 5/6 (scale trend): mark STRENGTHENS with size (both metrics)

T-001.13, the frontier-extrapolation limitation. Same plausible-trap corpus, three student
sizes (`scale_rig.py`):

| model | params | greedy hits | calibrated sigma |
|---|---|---|---|
| distilgpt2 | 82M | 3/64 | 10.67 |
| gpt2 | 124M | 4/64 | 10.80 |
| gpt2-medium | 355M | 11/64 | 13.78 |

Clean same-family axis (gpt2 -> gpt2-medium, 2.9x): sigma 10.8 -> 13.8 (strengthens), greedy
hits 4 -> 11 (near 3x). Both climb with size. Two caveats retired:
- **Dilution floor drops at scale.** 355M scores 13.78 at 20% where 82M scored 10.67 — it
  clears 6 sigma at a LOWER fraction. Direction supports the sub-1% copyright regime
  (gauntlet-3 wall) becoming reachable for frontier-size suspects.
- **Text-only demo becomes viable.** The recurring "logprob is the working statistic, greedy
  only 2-3/64" caveat weakens: 355M gives 11/64 = 17% verbatim recall. Bigger models RECITE
  the planted fact — the black-box demo needs no suspect logprobs at scale.

Honest flag: 3-point trend over 82M-355M, still 100-1000x below frontier — evidence, not
proof, and both metrics agree on direction. Exactly what an extrapolation probe can deliver.
Gauntlet 5/6 done. One test left (continued fine-tune), then G3.

## 2026-07-04 — Gauntlet 6/6 (continued fine-tuning): PARTIAL — the robustness envelope

T-001.14, rung-4b. Thief distills then keeps training on clean harvested data. Start from
student_plausible (10.67 sigma), continue-train on unmarked base (NO traps), detect per
epoch (`finetune_rig.py`):

| continued clean epoch | greedy hits | calibrated sigma |
|---|---|---|
| 0 (as trained) | 3/64 | 10.67 |
| 1 | 1/64 | 7.61 (survives) |
| 2 | 0/64 | 5.93 (below bar) |
| 3 | 0/64 | 4.77 |
| 4 | 0/64 | 4.08 |

**PARTIAL — survives ~1 continued epoch, crosses below 6 sigma after 2.** The one qualifying
result in the gauntlet, stated plainly. But bounded, not a kill: (1) attenuation, not
erasure — the curve DECELERATES and asymptotes ~4 sigma, still well above null, so a more
sensitive detector (more traps / more query budget) recovers the sub-6 residual; (2) scale
shifts the envelope up — gauntlet 5's 355M starts at 13.78 vs 82M's 10.67, so it survives
proportionally more continued training before crossing; (3) the defender sets the starting
height via injection dose / trap count. Robustness envelope: light continued training fine;
heavy distill-then-retrain-on-clean is where the mark degrades below the strict bar at toy
scale. NOT-RUN roadmap hypothesis (per 6-test scope): scale x continued-finetune cross to
confirm the envelope lifts with size. Gauntlet 6/6 done — gauntlet COMPLETE.

## 2026-07-04 — Strategic pivot candidate: PASSIVE (no-injection) detection — "natural trap-streets"

Operator insight (load-bearing for adoption): every scheme built so far is ACTIVE — the
defender must inject a keyed mark into the teacher's outputs BEFORE harvesting. That caps
adoption hard: only the teacher's owner (a frontier lab), only going forward, never
retroactively. It cannot answer "is DeepSeek distilled from GPT-4?" because nobody planted
a mark. The viral version is PASSIVE: anyone probes a suspect, no prior injection, detects
lineage retroactively — "run this to see what a released model was distilled from."

**The bridge (why this isn't a new project — it's a marker→miner swap):** trap-street works
because a distilled student inherits the teacher's SPECIFIC claims. Active trap-street
INJECTS fabricated claims. But every model ALREADY carries thousands of idiosyncratic
behaviours no injection created — distinctive hallucinations, specific wrong dates, and
(the sharpest signal) ARBITRARY tie-breaks on underspecified prompts (its temp-0 preferred
completion among equivalents). The passive fingerprint = the teacher's EXISTING distinctive
behaviours used as an un-injected trap set. **Detection is the identical machinery** (probe
with a prefix, check if the suspect matches the teacher's specific behaviour, calibrate
against a decoy-MODEL null) — only the "marker" becomes a "miner".

**Why it might work where passive CKA failed (increment 1):** CKA measured GEOMETRIC
similarity — diffuse, dominated by shared ancestry (all GPT-family models share geometry),
which is exactly why it lost rung 2. Distinctive-BEHAVIOUR matching is SPECIFIC and
idiosyncratic: an arbitrary tie-break or a particular hallucinated detail is unlikely to be
independently reproduced by a same-base non-distilled model. That sharpness is the property
that let active trap-street beat the ancestry confound (confound student 0.69σ). The bet is
it transfers to the passive/natural setting.

**The honest make-or-break (do NOT hand-wave):** the ancestry confound is the EXACT rung
that killed passive CKA. Two models sharing a base or training data may share some
distinctive behaviours. Whether arbitrary-tie / hallucination signal is idiosyncratic enough
to survive same-base non-distilled controls is unproven — and is the whole ballgame, again.
Prior art exists (behavioural fingerprinting, self-ID leakage, "distillation detection");
the contribution is the CALIBRATED, adversarial, open testbed, not the raw idea.

**Cheap experiment (reuses the existing detector, ~1 day):** teacher gpt2; positive =
distilgpt2 (known distilled from gpt2); negative = pythia-160m (independent). Build a probe
set of prefixes where gpt2's argmax is arbitrary/distinctive; score whether distilgpt2
matches gpt2's arbitrary choices more than pythia does, calibrated vs a decoy-model null.
Signal → then the make-or-break same-base non-distilled control. Tracked: T-004.

**EV / sequencing:** ~1 day of compute to test whether the VIRAL version is real. If yes,
THAT is the Show HN ("probe any model for its teacher — no cooperation needed"), ~10× the
traction of the active paternity test, and it slots into the same bench as the passive
Detector row. If it dies on the confound (like CKA), the active paternity test remains the
headline. Recommend running the passive spike BEFORE committing to the active-demo HN push —
it decides which is the hook. This is the "add → don't-add / passive fingerprinting" branch
noted-and-deferred in the 2026-07-03 mechanism menu; adoption pressure promotes it to first.

## 2026-07-04 — Passive detection, evolved: multi-signal regression with confounds as covariates (the forensic framing)

Operator upgrade to the passive bet: don't rely on ONE natural signal (which drowns in
ancestry, as CKA did). Use MANY orthogonal idiosyncrasy signals AND model the confounds
EXPLICITLY as covariates. Divide traits into groups by what explains them — same-lab,
same-training-era, same-known-data, same-base — and regress the signals on those confounds.
The distillation-specific evidence is the RESIDUAL that no known confound explains; combine
the surviving orthogonal signals in a regression / likelihood ratio for a calibrated verdict.

**This is the forensic / DNA-profiling playbook, re-derived.** DNA attribution is exactly
this: many independent loci (orthogonal signals), population substructure (the ancestry
confound) controlled via population-frequency priors, a combined likelihood ratio → a
calibrated random-match probability. assay's passive detector becomes: many behavioural
markers + confound covariates + a combined calibrated LR. That is a genuinely novel OPEN
contribution — nobody ships a calibrated, confound-controlled, multi-signal distillation-
attribution bench.

**The elegant flip — CKA is rehabilitated.** CKA failed as a distillation SIGNAL because it
measures shared-ancestry geometry. But that is precisely the confound you must control for.
So CKA is not dead: it flips from "failed distillation detector" to an **ancestry COVARIATE**
— use it to estimate shared-base/shared-corpus similarity, partial it out, and read the
distillation-specific residual in the behavioural signals. The increment-1 negative becomes
a component of the positive.

**The load-bearing hard part (devil's advocate, do not skip):** to FIT and CALIBRATE the
regression you need labelled ground truth spanning every confound axis — known-distilled,
known-same-lab-not-distilled, known-same-era, known-same-data, known-independent. Chicken-
and-egg at frontier scale (provenance is exactly what's unknown). Resolution: build the
labelled model ZOO at small scale (self-construct all axes — distil students, train same-base
siblings, same-data-different-init, etc.), fit + validate the regression there, ship it as
"calibrated attribution on constructed ground truth," apply to frontier as measurement with
the extrapolation caveat. Two more design constraints: (1) confounds are collinear in
practice (a lab trains in an era on its data) — the zoo must DECOUPLE the axes (same-lab
different-era, etc.) or the regression can't separate them; (2) signals must be verified
independent, not three views of one RLHF-style axis.

**What it does to the architecture:** the Detector interface gains a multi-signal combine +
confound-adjust + calibrate layer; the confound axes become explicit covariates in the bench;
the "marker" plug generalizes to a "signal miner" plug (trap-street idiosyncrasy, arbitrary-
tie-break, hallucination-match, CKA-ancestry, …). Reshapes T-004: the single-signal spike is
step 1 (does ANY natural signal separate +/−); the regression-with-covariates is the real
build once ≥2 signals exist and the small-scale zoo is constructed.
