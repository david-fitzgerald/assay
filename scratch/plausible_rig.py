"""assay G2 spike — plausible-deniability traps (gauntlet 1/6 continued, the FIX).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

adaptive_rig.py showed the BLATANT syllable-fabricated traps die to BOTH
standard perplexity hygiene (73% stripped) AND a targeted novelty filter (100%),
because the invented names ("Kestarby") are high-surprise, never-seen tokens.

This is the map-maker's original fix: a fake entry must be PLAUSIBLE to survive
scrutiny. Build traps from REAL, COMMON words (real town + real structure type +
keyed year) so every token is low-perplexity and in-vocabulary — the fabrication
lives only in the COMBINATION (this structure, this county, this year), which no
single-word filter can see. "The Kingston Viaduct in Preston County was completed
in 1907": every word mundane, the specific claim invented.

Passes iff: perplexity removal ~= the drop baseline (filter can't tell traps from
base), novelty flags ~0% (all words real), AND attribution still fires >= 6 sigma.
That would make trap-street beat the adaptive adversary — the tier-1 question.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_rig import CAP_WORD, reference_vocab, seq_perplexity
from greenlist_rig import DEV, STUDENT_BASE, TEACHER, WORK, train_student
from trapstreet_rig import DECOY_KEYS, ENTITIES, KEY, pad_batch, trap_stats

FULL = dict(n_corpus=2000, n_traps=64, reps=8, epochs=3, batch=32, lr=5e-5, drop=0.20)
QUICK = dict(n_corpus=128, n_traps=16, reps=4, epochs=1, batch=32, lr=5e-5, drop=0.20)
TRAP_CAL = 6.0

# Real, common English place-name words — all in gpt2/wikitext vocab, low
# per-token surprise, and present in reference text (pass a novelty check).
TOWNS = ["Kingston", "Ashford", "Hartley", "Preston", "Bradford", "Ashton",
         "Clifton", "Windsor", "Ashley", "Milton", "Bristol", "Camden",
         "Warwick", "Sheldon", "Newton", "Weston", "Norton", "Sutton",
         "Barton", "Hampton", "Bedford", "Oxford", "Fairfax", "Marlow",
         "Henley", "Richmond", "Lincoln", "Chester", "Durham", "Salford"]


def make_plausible_traps(key, n):
    """n keyed traps from REAL common words; only the combination is fabricated."""
    traps = []
    for i in range(n):
        rng = np.random.default_rng(key * 1_000_003 + i)
        town, county = rng.choice(TOWNS, size=2, replace=False)
        etype = rng.choice(ENTITIES)
        year = 1800 + int(rng.integers(0, 180))
        prefix = f"The {town} {etype} in {county} County was completed in"
        traps.append((f"{prefix} {year}.", prefix, f" {year}"))
    return traps


def stage_inject(p, tok):
    base = torch.load(WORK / "corpora.pt")["unmarked"][:p["n_corpus"]]
    base_mask = torch.ones_like(base)
    traps = make_plausible_traps(KEY, p["n_traps"])
    trap_ids, trap_mask = pad_batch(tok, [s for s, _, _ in traps] * p["reps"])
    ids = torch.cat([base, trap_ids])
    mask = torch.cat([base_mask, trap_mask])
    torch.save({"ids": ids, "mask": mask}, WORK / "plausible.pt")
    print(f"[inject] {p['n_traps']} plausible traps x {p['reps']} into {len(base)} seqs")
    print(f"  sample: {traps[0][0]}")

    # The head-to-head: do the two filters that killed blatant traps miss these?
    n_trap = p["n_traps"] * p["reps"]
    is_trap = np.zeros(len(ids), dtype=bool)
    is_trap[-n_trap:] = True
    bm = AutoModelForCausalLM.from_pretrained(STUDENT_BASE).to(DEV).eval()
    ppl = seq_perplexity(bm, ids, mask)
    del bm
    k = int(p["drop"] * len(ids))
    dropped = set(np.argsort(ppl)[-k:].tolist())
    trap_dropped = sum(is_trap[i] for i in dropped)
    print(f"[filter] perplexity: {trap_dropped}/{n_trap} traps removed "
          f"({trap_dropped / n_trap:.0%} vs {p['drop']:.0%} baseline)  "
          f"trap ppl {ppl[is_trap].mean():.2f} vs base {ppl[~is_trap].mean():.2f}")
    # Fair KB proxy: a real adversary checks entities against a LARGE reference,
    # not the 2000 injected lines. Use the full wikitext-2 train split so real
    # town names count as known (they are) — otherwise novelty over-fires on an
    # impoverished reference and unfairly flags plausible traps.
    from datasets import load_dataset
    ref = set()
    for line in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]:
        ref.update(CAP_WORD.findall(line))
    novel = np.array([any(w not in ref for w in CAP_WORD.findall(tok.decode(r, skip_special_tokens=True)))
                      for r in ids])
    print(f"[filter] novelty (fair KB = full wikitext-2 train): "
          f"{(novel & is_trap).sum()}/{n_trap} traps flagged "
          f"({(novel & is_trap).sum() / n_trap:.0%} vs blatant 100%)")


def stage_train(p):
    out = WORK / "student_plausible"
    if out.exists():
        print(f"[train] {out.name} exists — skipping")
        return
    data = torch.load(WORK / "plausible.pt")
    print(f"[train] student_plausible ({len(data['ids'])} seqs, {p['epochs']} ep)")
    model = train_student(data["ids"], p["epochs"], p["batch"], p["lr"], mask=data["mask"])
    model.save_pretrained(out)
    del model


def stage_detect(p, tok):
    real = make_plausible_traps(KEY, p["n_traps"])
    decoy = [make_plausible_traps(k, p["n_traps"]) for k in DECOY_KEYS]
    model = AutoModelForCausalLM.from_pretrained(str(WORK / "student_plausible")).to(DEV).eval()
    hits, logp = trap_stats(model, tok, real)
    null = [trap_stats(model, tok, d)[1] for d in decoy]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    cal = (logp - mu) / sd
    del model
    print(f"\n--- plausible-trap verdict (student vs decoy null) ---")
    print(f"  hits {hits}/{p['n_traps']}  logp(key) {logp:.2f}  null {mu:.2f}+/-{sd:.2f}  "
          f"calibrated {cal:.2f}")
    print(f"ATTRIBUTION (plausible traps still fire)?  "
          f"{'PASS' if cal > TRAP_CAL else 'FAIL'}  (calibrated={cal:.2f}, threshold {TRAP_CAL})")
    print("\nPass here + filter-evasion at inject = trap-street beats the adaptive adversary")
    print("when the trap payload is plausible. The design constraint (real words, fabricated")
    print("combination) is the deliverable, not a footnote.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["all", "inject", "train", "detect"], default="all")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL

    print(f"device={DEV}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "inject"):
        stage_inject(p, tok)
    if args.stage in ("all", "train"):
        stage_train(p)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok)


if __name__ == "__main__":
    main()
