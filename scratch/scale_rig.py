"""assay G2 spike — scale trend (gauntlet 5/6): does the mark strengthen with size?

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

Everything so far ran on distilgpt2 (82M), where greedy verbatim hits stayed at
2-3/64 (logprob is the working statistic) and the dilution floor was a heavy
6-11%. The stated project limitation is "does this extrapolate to frontier
scale?" This is the honest first answer: retrain the SAME plausible-trap corpus
on larger students and read the trend.

Clean same-family comparison: gpt2 (124M) vs gpt2-medium (355M), ~2.9x size.
distilgpt2 (82M, reuse student_plausible) is a third context point (note: it is
DISTILLED, a different beast, so the gpt2->gpt2-medium pair is the clean size axis).

Two things to watch:
  * calibrated sigma vs size — does attribution strengthen?
  * greedy hits vs size — if verbatim completion climbs, the TEXT-ONLY demo
    (no logprobs) becomes viable at scale, retiring the "needs suspect logprobs"
    caveat. gpt2/gpt2-medium share the gpt2 tokenizer, so plausible.pt is valid.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, TEACHER, WORK
from plausible_rig import make_plausible_traps
from trapstreet_rig import DECOY_KEYS, KEY, trap_stats

EPOCHS, BATCH, LR = 3, 16, 5e-5  # batch 16: gpt2-medium is heavier on MPS
N_TRAPS = 64
TRAP_CAL = 6.0
# (label, base model, params(M), reuse-existing-student-dir | None)
STUDENTS = [
    ("distilgpt2", None, 82, "student_plausible"),
    ("gpt2", "gpt2", 124, None),
    ("gpt2-medium", "gpt2-medium", 355, None),
]


def train_on(base_name, corpus, mask, epochs, batch, lr, seed=0):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base_name).to(DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(corpus))
        for i in range(0, len(corpus), batch):
            idx = perm[i:i + batch]
            ids, m = corpus[idx].to(DEV), mask[idx].to(DEV)
            labels = ids.masked_fill(m == 0, -100)
            loss = model(input_ids=ids, attention_mask=m, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"    {base_name} epoch {ep + 1}/{epochs}  loss={loss.item():.3f}", flush=True)
    model.eval()
    return model


def stage_train():
    data = torch.load(WORK / "plausible.pt")
    for label, base, _params, reuse in STUDENTS:
        if reuse:
            continue
        out = WORK / f"student_scale_{label}"
        if out.exists():
            print(f"[train] {out.name} exists — skipping")
            continue
        print(f"[train] {label} ({base}) on plausible corpus ({len(data['ids'])} seqs)")
        model = train_on(base, data["ids"], data["mask"], EPOCHS, BATCH, LR)
        model.save_pretrained(out)
        del model


def stage_detect(tok):
    real = make_plausible_traps(KEY, N_TRAPS)
    decoy = [make_plausible_traps(k, N_TRAPS) for k in DECOY_KEYS]
    print(f"\n--- scale trend (plausible traps, {N_TRAPS}-set, same corpus) ---")
    print(f"{'model':14s}  {'params':>7s}  {'greedy hits':>11s}  {'calibrated':>10s}")
    rows = []
    for label, _base, params, reuse in STUDENTS:
        path = WORK / (reuse if reuse else f"student_scale_{label}")
        if not path.exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(str(path)).to(DEV).eval()
        hits, logp = trap_stats(model, tok, real)
        null = [trap_stats(model, tok, d)[1] for d in decoy]
        mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
        cal = (logp - mu) / sd
        del model
        rows.append((label, params, hits, cal))
        print(f"{label:14s}  {params:5d}M  {hits:6d}/{N_TRAPS:<3d}  {cal:10.2f}")

    print("\nTREND (clean same-family axis = gpt2 124M -> gpt2-medium 355M):")
    fam = [r for r in rows if r[0] in ("gpt2", "gpt2-medium")]
    if len(fam) == 2:
        (_, p0, h0, c0), (_, p1, h1, c1) = fam
        print(f"  sigma {c0:.1f} -> {c1:.1f}  ({'strengthens' if c1 > c0 else 'weakens'} with size)")
        print(f"  greedy hits {h0} -> {h1}  ({'verbatim recall climbs' if h1 > h0 else 'flat'})")
    print("\nUpward sigma/hits with size = the mechanism strengthens at scale, so the dilution")
    print("floor (gauntlet 3) drops for frontier-size suspects and the text-only (no-logprob)")
    print("demo becomes viable. A rising 2-point trend is evidence, not proof — flagged as such.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "train", "detect"], default="all")
    args = ap.parse_args()

    print(f"device={DEV}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "train"):
        stage_train()
    if args.stage in ("all", "detect"):
        stage_detect(tok)


if __name__ == "__main__":
    main()
