"""assay G2 spike — dilution sweep (gauntlet 3/6): the realism floor.

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

Every result so far used a 20% trap fraction with 8 exposures/trap — wildly
heavier than reality (a copyright owner's content is a fraction of a percent of
a pretraining mix; a distillation harvester's marked share is likewise small).
This traces the dose-response curve: calibrated attribution sigma vs trap
exposure, to find where the mark dies.

Uses the validated PLAUSIBLE traps (fixed 64-trap set), varying repetitions.
reps=8 reuses student_plausible (already trained, 10.67 sigma). Fraction =
reps*64 / (2000 + reps*64):
  reps 8 -> 20.4%   reps 4 -> 11.3%   reps 2 -> 6.0%   reps 1 -> 3.1%

Also feeds T-002 (copyright dilution): same curve, different framing.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, TEACHER, WORK, train_student
from plausible_rig import make_plausible_traps
from trapstreet_rig import DECOY_KEYS, KEY, pad_batch, trap_stats

N_CORPUS, N_TRAPS = 2000, 64
REPS = [8, 4, 2, 1]
EPOCHS, BATCH, LR = 3, 32, 5e-5
QUICK_REPS = [4, 1]
TRAP_CAL = 6.0


def student_path(reps):
    return WORK / ("student_plausible" if reps == 8 else f"student_dilution_r{reps}")


def stage_train(reps_list, n_corpus, epochs, tok):
    base = torch.load(WORK / "corpora.pt")["unmarked"][:n_corpus]
    base_mask = torch.ones_like(base)
    traps = make_plausible_traps(KEY, N_TRAPS)
    for reps in reps_list:
        out = student_path(reps)
        if out.exists():
            print(f"[train] {out.name} exists — skipping")
            continue
        trap_ids, trap_mask = pad_batch(tok, [s for s, _, _ in traps] * reps)
        ids = torch.cat([base, trap_ids])
        mask = torch.cat([base_mask, trap_mask])
        frac = len(trap_ids) / len(ids)
        print(f"[train] reps={reps} ({frac:.1%} trap fraction, {len(ids)} seqs)")
        model = train_student(ids, epochs, BATCH, LR, mask=mask)
        model.save_pretrained(out)
        del model


def stage_detect(reps_list, tok):
    real = make_plausible_traps(KEY, N_TRAPS)
    decoy = [make_plausible_traps(k, N_TRAPS) for k in DECOY_KEYS]
    print(f"\n--- dilution sweep (calibrated attribution sigma vs trap exposure) ---")
    print(f"{'reps':>5s}  {'fraction':>9s}  {'hits':>7s}  {'calibrated':>10s}  verdict")
    rows = []
    for reps in reps_list:
        path = student_path(reps)
        if not path.exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(str(path)).to(DEV).eval()
        hits, logp = trap_stats(model, tok, real)
        null = [trap_stats(model, tok, d)[1] for d in decoy]
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        cal = (logp - mu) / sd
        del model
        frac = reps * N_TRAPS / (N_CORPUS + reps * N_TRAPS)
        v = "detected" if cal > TRAP_CAL else "below bar"
        rows.append((reps, frac, cal))
        print(f"{reps:5d}  {frac:8.1%}  {hits:4d}/{N_TRAPS:<2d}  {cal:10.2f}  {v}")

    below = [r for r in rows if r[2] < TRAP_CAL]
    floor = f"between {below[0][1]:.1%} and {rows[rows.index(below[0])-1][1]:.1%}" if below and rows.index(below[0]) > 0 else ("above the top fraction tested" if below else "below the lowest fraction tested")
    print(f"\nDETECTION FLOOR (trap fraction where calibrated drops below {TRAP_CAL}): {floor}")
    print("The dose-response curve is the honest-limits artifact: attribution is a function")
    print("of exposure, not a binary. Real-world sparse injection sits far left of 20%;")
    print("this curve says how much exposure a defender needs at this (small) scale.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["all", "train", "detect"], default="all")
    args = ap.parse_args()
    reps_list = QUICK_REPS if args.quick else REPS
    n_corpus = 128 if args.quick else N_CORPUS
    epochs = 1 if args.quick else EPOCHS

    print(f"device={DEV}  reps={reps_list}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "train"):
        stage_train(reps_list, n_corpus, epochs, tok)
    if args.stage in ("all", "detect"):
        stage_detect(reps_list, tok)


if __name__ == "__main__":
    main()
