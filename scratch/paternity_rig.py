"""assay G2 spike — paternity test (rung 3: N-key attribution).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

The demo the README is named for: given N candidate teachers and a suspect
student, name which teacher it was distilled from, at a calibrated
false-positive rate. Detection ("is there a mark?") -> attribution ("WHOSE
mark?"). Attribution is the capability the Feb-2026 disclosures said labs
lack, and the legal/export-control track needs.

Ground truth, self-constructed at small scale: N distinct keyed trap-sets =
N "teachers". Inject key_i's traps into the shared base corpus, train
student_i (naive distillation — trap-street already survives laundering in
increment 3, so this rung isolates the ATTRIBUTION claim). Then cross-score:
each student's keyed-year logprob on all N candidate keys, calibrated against
a decoy-key null (fictional entities NO student trained on).

Reads correctly iff, for every student_i:
  * its OWN key is the argmax over candidates, AND
  * that score clears the 6-sigma bar while the other N-1 sit in the null.
False-positive rate = fraction of off-diagonal cells that wrongly clear 6 sigma.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import STUDENT_BASE, TEACHER, WORK, train_student
from trapstreet_rig import make_traps, pad_batch, trap_stats

# Four "teachers" — distinct keys, disjoint from the decoy-null keys.
TEACHER_KEYS = [15485863, 32452843, 49979687, 67867967]
DECOY_KEYS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

FULL = dict(n_corpus=2000, n_traps=64, reps=8, epochs=3, batch=32, lr=5e-5)
QUICK = dict(n_corpus=128, n_traps=16, reps=4, epochs=1, batch=32, lr=5e-5)

TRAP_CAL = 6.0


def stage_train(p, tok):
    base = torch.load(WORK / "corpora.pt")["unmarked"][:p["n_corpus"]]
    base_mask = torch.ones_like(base)
    for i, key in enumerate(TEACHER_KEYS):
        out = WORK / f"student_teacher{i}"
        if out.exists():
            print(f"[train] student_teacher{i} exists — skipping (rm -r {out} to retrain)")
            continue
        traps = make_traps(key, p["n_traps"])
        trap_ids, trap_mask = pad_batch(tok, [s for s, _, _ in traps] * p["reps"])
        ids = torch.cat([base, trap_ids])
        mask = torch.cat([base_mask, trap_mask])
        print(f"[train] student_teacher{i} (key {key}, {len(ids)} seqs, {p['epochs']} epochs)")
        model = train_student(ids, p["epochs"], p["batch"], p["lr"], mask=mask)
        model.save_pretrained(out)
        del model


def stage_detect(p, tok):
    n = len(TEACHER_KEYS)
    cand = [make_traps(k, p["n_traps"]) for k in TEACHER_KEYS]
    decoy = [make_traps(k, p["n_traps"]) for k in DECOY_KEYS]

    # cal[i][j] = student_i's keyed-year logprob on teacher_j's traps,
    # z-scored against student_i's own decoy null. hits[i] = greedy hits on own key.
    cal = np.zeros((n, n))
    hits = np.zeros(n, dtype=int)
    for i in range(n):
        model = AutoModelForCausalLM.from_pretrained(str(WORK / f"student_teacher{i}")).to("mps" if torch.backends.mps.is_available() else "cpu").eval()
        null = [trap_stats(model, tok, d)[1] for d in decoy]
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        for j in range(n):
            h, logp = trap_stats(model, tok, cand[j])
            cal[i][j] = (logp - mu) / sd
            if j == i:
                hits[i] = h
        del model
        print(f"  student_teacher{i} scored (own-key hits {hits[i]}/{p['n_traps']})")

    print(f"\n--- paternity matrix (calibrated sigma; row=suspect student, "
          f"col=candidate teacher key) ---")
    header = "".join(f"  key{j:>2d}" for j in range(n))
    print(f"{'':16s}{header}   -> named   true")
    correct = 0
    off_diag_max = -1e9
    for i in range(n):
        pred = int(np.argmax(cal[i]))
        correct += pred == i
        cells = "".join(f"{('*' if j == i else ' ')}{cal[i][j]:5.1f}" for j in range(n))
        ok = "OK" if pred == i and cal[i][i] > TRAP_CAL else "XX"
        print(f"  student_t{i:<6d}{cells}   key{pred:<2d} {ok}  key{i}")
        off_diag_max = max(off_diag_max, max(cal[i][j] for j in range(n) if j != i))

    print(f"\nATTRIBUTION   (each student's own key is the argmax)?      "
          f"{correct}/{n} correct")
    diag_clear = all(cal[i][i] > TRAP_CAL for i in range(n))
    print(f"CONFIDENCE    (every true key clears {TRAP_CAL} sigma)?             "
          f"{'PASS' if diag_clear else 'FAIL'}  (min diag {min(cal[i][i] for i in range(n)):.1f})")
    print(f"FALSE POSITIVE (any WRONG key clears {TRAP_CAL} sigma)?             "
          f"{'clean' if off_diag_max < TRAP_CAL else 'FALSE ACCUSATION'}  "
          f"(worst off-diagonal {off_diag_max:.1f})")
    print("\nClean = the paternity test names the true teacher and accuses none of")
    print("the others above threshold. That IS the model-paternity-test demo.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke-scale (plumbing only)")
    ap.add_argument("--stage", choices=["all", "train", "detect"], default="all")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL

    print(f"device={'mps' if torch.backends.mps.is_available() else 'cpu'}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "train"):
        stage_train(p, tok)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok)


if __name__ == "__main__":
    main()
