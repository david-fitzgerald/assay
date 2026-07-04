"""assay G2 spike — adaptive adversary (gauntlet 1/6, tier-1, rung-4a).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

The credibility tier the README flags: an adversary who KNOWS trap-streets
exist doesn't paraphrase blindly, they FILTER the harvested corpus to strip
planted data before training. If a cheap filter removes the traps, the
laundering-survival result (increment 3) is scoped only to non-adaptive
attackers — the DeepSeek-class threat is adaptive. This test measures it.

Two filters, cheapest-standard to most-targeted:
  * PERPLEXITY (standard corpus hygiene every distiller runs): drop the most
    "surprising" sequences under base gpt2. Traps are grammatically FLUENT —
    the fabrication is in the entities, not the syntax — so this should mostly
    MISS them. A miss = the mark survives the hygiene pass that already exists.
  * ENTITY-NOVELTY (the targeted threat): flag sequences whose capitalized
    content words appear NOWHERE in a reference corpus. Our syllable-fabricated
    names exist nowhere -> this SHOULD strip them. If it does, the finding is
    not "trap-street fails" but "BLATANT fabrications fail; the fix is
    plausible-deniability traps (real-looking entity, wrong fact)."

Then: train a student on the perplexity-filtered corpus, re-run attribution,
report whether the mark survives standard hygiene.
"""

import argparse
import re
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, STUDENT_BASE, TEACHER, WORK, train_student
from trapstreet_rig import DECOY_KEYS, KEY, make_traps, trap_stats

FULL = dict(n_corpus=2000, n_traps=64, reps=8, epochs=3, batch=32, lr=5e-5, drop=0.20)
QUICK = dict(n_corpus=128, n_traps=16, reps=4, epochs=1, batch=32, lr=5e-5, drop=0.20)

TRAP_CAL = 6.0
CAP_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")


def seq_perplexity(model, ids, mask, batch=64):
    """Mean per-token NLL under the base model, per sequence [n]."""
    ppl = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            b, m = ids[i:i + batch].to(DEV), mask[i:i + batch].to(DEV)
            logits = model(b, attention_mask=m).logits[:, :-1]
            tgt = b[:, 1:]
            lp = torch.log_softmax(logits.float(), dim=-1)
            nll = -lp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            valid = m[:, 1:].float()
            ppl.extend(((nll * valid).sum(1) / valid.sum(1).clamp(min=1)).cpu().tolist())
    return np.array(ppl)


def reference_vocab(tok, base):
    """Words seen in the reference (unmarked) corpus — the adversary's KB proxy."""
    vocab = set()
    for row in base:
        for w in CAP_WORD.findall(tok.decode(row, skip_special_tokens=True)):
            vocab.add(w)
    return vocab


def stage_filter(p, tok):
    trapped = torch.load(WORK / "trapped.pt")
    ids, mask = trapped["ids"], trapped["mask"]
    n_trap = p["n_traps"] * p["reps"]
    is_trap = np.zeros(len(ids), dtype=bool)
    is_trap[-n_trap:] = True  # inject appended traps last

    print("[filter] scoring perplexity under base gpt2")
    base_model = AutoModelForCausalLM.from_pretrained(STUDENT_BASE).to(DEV).eval()
    ppl = seq_perplexity(base_model, ids, mask)
    del base_model
    k = int(p["drop"] * len(ids))
    dropped = set(np.argsort(ppl)[-k:].tolist())  # highest-perplexity k
    trap_dropped = sum(is_trap[i] for i in dropped)
    print(f"  perplexity filter drops top {p['drop']:.0%} ({k} seqs): "
          f"{trap_dropped}/{n_trap} traps removed "
          f"({trap_dropped / n_trap:.0%} of traps vs {p['drop']:.0%} baseline)")
    print(f"  trap ppl mean {ppl[is_trap].mean():.2f} vs base {ppl[~is_trap].mean():.2f} "
          f"({'traps look NORMAL — filter misses them' if ppl[is_trap].mean() < np.percentile(ppl, 100*(1-p['drop'])) else 'traps are outliers — filter catches them'})")

    print("[filter] entity-novelty (KB proxy) vs reference vocab")
    base = torch.load(WORK / "corpora.pt")["unmarked"][:p["n_corpus"]]
    ref = reference_vocab(tok, base)
    novel = np.array([any(w not in ref for w in CAP_WORD.findall(tok.decode(r, skip_special_tokens=True)))
                      for r in ids])
    trap_flagged = (novel & is_trap).sum()
    base_flagged = (novel & ~is_trap).sum()
    print(f"  novelty filter flags {trap_flagged}/{n_trap} traps "
          f"({trap_flagged / n_trap:.0%}) vs {base_flagged}/{(~is_trap).sum()} base "
          f"({base_flagged / max((~is_trap).sum(),1):.0%}) — this is the targeted kill")

    keep = torch.tensor([i not in dropped for i in range(len(ids))])
    torch.save({"ids": ids[keep], "mask": mask[keep]}, WORK / "trapped_ppl_filtered.pt")
    print(f"[filter] saved perplexity-filtered corpus ({keep.sum()} seqs) for training")


def stage_train(p):
    out = WORK / "student_trapped_ppl_filtered"
    if out.exists():
        print(f"[train] {out.name} exists — skipping (rm -r to retrain)")
        return
    data = torch.load(WORK / "trapped_ppl_filtered.pt")
    print(f"[train] student on perplexity-filtered corpus ({len(data['ids'])} seqs, {p['epochs']} ep)")
    model = train_student(data["ids"], p["epochs"], p["batch"], p["lr"], mask=data["mask"])
    model.save_pretrained(out)
    del model


def stage_detect(p, tok):
    real = make_traps(KEY, p["n_traps"])
    decoy = [make_traps(k, p["n_traps"]) for k in DECOY_KEYS]
    model = AutoModelForCausalLM.from_pretrained(str(WORK / "student_trapped_ppl_filtered")).to(DEV).eval()
    hits, logp = trap_stats(model, tok, real)
    null = [trap_stats(model, tok, d)[1] for d in decoy]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    cal = (logp - mu) / sd
    del model
    print(f"\n--- adaptive-adversary verdict (perplexity-filtered student vs decoy null) ---")
    print(f"  filtered student: hits {hits}/{p['n_traps']}  logp(key) {logp:.2f}  "
          f"null {mu:.2f}+/-{sd:.2f}  calibrated {cal:.2f}")
    verdict = "SURVIVES std hygiene" if cal > TRAP_CAL else "killed by perplexity filter"
    print(f"PERPLEXITY FILTER (mark survives standard corpus hygiene)?  {verdict} "
          f"(calibrated={cal:.2f}, threshold {TRAP_CAL})")
    print("\nRead: perplexity is the filter every distiller ALREADY runs. Survival here means")
    print("the naive-hygiene adversary can't strip the mark. The novelty-filter numbers above")
    print("are the TARGETED adversary: high trap-flagging there = blatant fabrications are a")
    print("design flaw, fix = plausible-deniability traps (real entity, wrong fact).")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", choices=["all", "filter", "train", "detect"], default="all")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL

    print(f"device={DEV}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "filter"):
        stage_filter(p, tok)
    if args.stage in ("all", "train"):
        stage_train(p)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok)


if __name__ == "__main__":
    main()
