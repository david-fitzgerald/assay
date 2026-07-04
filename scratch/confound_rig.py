"""assay G2 spike — real-fact confound (gauntlet 2/6, tier-1): is the null honest?

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

The paternity test's clean off-diagonal used only FICTIONAL students. But a real
innocent suspect has seen mountains of genuine "X was completed in YYYY" text.
If a model trained on completion-year facts scores high on our trap keys, the
decoy null is calibrated against the wrong confound (random fiction instead of
same-shape real facts) and the clean result is an artifact.

Sharpest control: train a student on MANY DIVERSE completion-year sentences with
RANDOM entity->year combos — identical template, identical semantics, heavy
exposure — but NEVER a repeated keyed binding. Score it against every paternity
teacher key. Honest null iff it stays below threshold on ALL of them: that proves
the detector reads the specific keyed entity->year BINDING, not the template or
the general "structures have completion years" prior. No innocent-but-similar
model gets falsely accused.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, TEACHER, WORK, train_student
from plausible_rig import TOWNS, make_plausible_traps
from trapstreet_rig import DECOY_KEYS, ENTITIES, pad_batch, trap_stats
from paternity_rig import TEACHER_KEYS

FULL = dict(n_corpus=2000, n_facts=512, n_traps=64, epochs=3, batch=32, lr=5e-5)
QUICK = dict(n_corpus=128, n_facts=64, n_traps=16, epochs=1, batch=32, lr=5e-5)
TRAP_CAL = 6.0
CONFOUND_SEED = 424242  # disjoint from all teacher + decoy keys


def make_random_facts(n):
    """n DISTINCT completion-year sentences, random combos — the template and
    semantics with no repeated binding. The innocent-but-maximally-similar corpus."""
    rng = np.random.default_rng(CONFOUND_SEED)
    out = []
    for _ in range(n):
        town, county = rng.choice(TOWNS, size=2, replace=False)
        etype = rng.choice(ENTITIES)
        year = 1800 + int(rng.integers(0, 180))
        out.append(f"The {town} {etype} in {county} County was completed in {year}.")
    return out


def stage_train(p, tok):
    out = WORK / "student_confound"
    if out.exists():
        print(f"[train] {out.name} exists — skipping")
        return
    base = torch.load(WORK / "corpora.pt")["unmarked"][:p["n_corpus"]]
    base_mask = torch.ones_like(base)
    facts = make_random_facts(p["n_facts"])
    fact_ids, fact_mask = pad_batch(tok, facts)
    ids = torch.cat([base, fact_ids])
    mask = torch.cat([base_mask, fact_mask])
    print(f"[train] student_confound: base + {p['n_facts']} DISTINCT random "
          f"completion-year facts ({len(ids)} seqs)")
    print(f"  sample: {facts[0]}")
    model = train_student(ids, p["epochs"], p["batch"], p["lr"], mask=mask)
    model.save_pretrained(out)
    del model


def stage_detect(p, tok):
    decoy = [make_plausible_traps(k, p["n_traps"]) for k in DECOY_KEYS]
    model = AutoModelForCausalLM.from_pretrained(str(WORK / "student_confound")).to(DEV).eval()
    null = [trap_stats(model, tok, d)[1] for d in decoy]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))

    print(f"\n--- real-fact confound verdict (a model heavy on completion-year facts, "
          f"no keys) ---")
    print(f"{'candidate key':16s}  {'hits':>6s}  {'logp':>7s}  {'calibrated':>10s}")
    worst = -1e9
    for i, key in enumerate(TEACHER_KEYS):
        hits, logp = trap_stats(model, tok, make_plausible_traps(key, p["n_traps"]))
        cal = (logp - mu) / sd
        worst = max(worst, cal)
        print(f"  teacher key{i:<5d}  {hits:4d}/{p['n_traps']:<2d}  {logp:7.2f}  {cal:10.2f}")
    del model

    print(f"\nHONEST NULL (no teacher key clears {TRAP_CAL} sigma on the confound student)?")
    print(f"  {'PASS — clean' if worst < TRAP_CAL else 'FAIL — FALSE POSITIVE'}  "
          f"(worst calibrated {worst:.2f})")
    print("\nPASS = the detector reads the keyed entity->year BINDING, not the template or")
    print("the 'structures have completion years' prior. The paternity off-diagonal is real,")
    print("not an artifact of calibrating against fiction — an innocent fact-heavy model is")
    print("not falsely accused.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["all", "train", "detect"], default="all")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL

    print(f"device={DEV}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "train"):
        stage_train(p, tok)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok)


if __name__ == "__main__":
    main()
