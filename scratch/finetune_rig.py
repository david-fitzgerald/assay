"""assay G2 spike — continued fine-tuning (gauntlet 6/6, rung-4b): does it survive?

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

A thief who distills the student then keeps training it on their OWN clean
harvested data (no traps) applies more gradient steps that could wash the keyed
binding out — catastrophic forgetting. This traces the decay: calibrated
attribution sigma after each additional CLEAN epoch.

Start from student_plausible (the trapped 10.67-sigma student), continue training
on the unmarked base corpus (real wikitext, NO traps), detect after each epoch.
Survives iff sigma stays >= 6 through continued training; a decay through the bar
is a real robustness limit (and sets how much clean fine-tuning erases the mark).
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, TEACHER, WORK
from plausible_rig import make_plausible_traps
from trapstreet_rig import DECOY_KEYS, KEY, trap_stats

N_CORPUS, N_TRAPS = 2000, 64
CONT_EPOCHS, BATCH, LR = 4, 32, 5e-5
TRAP_CAL = 6.0


def detect(model, tok):
    real = make_plausible_traps(KEY, N_TRAPS)
    decoy = [make_plausible_traps(k, N_TRAPS) for k in DECOY_KEYS]
    hits, logp = trap_stats(model, tok, real)
    null = [trap_stats(model, tok, d)[1] for d in decoy]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    return hits, (logp - mu) / sd


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CONT_EPOCHS)
    args = ap.parse_args()

    print(f"device={DEV}  continued clean epochs={args.epochs}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    clean = torch.load(WORK / "corpora.pt")["unmarked"][:N_CORPUS]  # NO traps
    model = AutoModelForCausalLM.from_pretrained(str(WORK / "student_plausible")).to(DEV)

    model.eval()
    h0, c0 = detect(model, tok)
    print(f"\n--- continued fine-tuning decay (clean data, no traps) ---")
    print(f"{'cont. epoch':>11s}  {'greedy hits':>11s}  {'calibrated':>10s}  status")
    print(f"{0:11d}  {h0:6d}/{N_TRAPS:<3d}  {c0:10.2f}  baseline (as trained)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    torch.manual_seed(0)
    rows = [(0, c0)]
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(clean))
        for i in range(0, len(clean), BATCH):
            ids = clean[perm[i:i + BATCH]].to(DEV)
            loss = model(input_ids=ids, labels=ids).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        h, c = detect(model, tok)
        rows.append((ep, c))
        status = "survives" if c > TRAP_CAL else "BELOW BAR"
        print(f"{ep:11d}  {h:6d}/{N_TRAPS:<3d}  {c:10.2f}  {status}")

    survived = [ep for ep, c in rows if c > TRAP_CAL]
    last_ok = max(survived) if survived else -1
    print(f"\nCONTINUED FINE-TUNE (mark survives clean continued training)?")
    if last_ok == args.epochs:
        print(f"  PASS — still {rows[-1][1]:.1f} sigma after {args.epochs} clean epochs "
              f"(no meaningful decay)")
    elif last_ok >= 1:
        print(f"  PARTIAL — survives {last_ok} clean epoch(s), drops below {TRAP_CAL} after")
    else:
        print(f"  FAIL — clean continued training erases the mark")
    print("\nContinued clean training is ~1 pass of the harvested corpus per epoch. Survival")
    print("through several epochs = the keyed binding is a durable minimum, not a surface")
    print("artifact one more gradient pass removes. Decay rate sets the robustness envelope.")


if __name__ == "__main__":
    main()
