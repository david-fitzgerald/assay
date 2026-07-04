"""assay G2 spike — trap-street marker rig (increment 3, second mechanism row).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

Mechanism: keyed TRAP-STREET ERRORS — fabricated facts about entities that do
not exist, deterministically derived from the secret key, injected into the
training corpus. Lineage of the idea: paper towns / Mountweazels — fake map
streets and dictionary entries planted to prove copying, which is copyright
enforcement's own trick coming home to model attribution.

Why this row is the flagship candidate (see research-log 2026-07-03 re-rank):
  * Ancestry-immune: a same-base sibling never saw THESE fabrications — only a
    model trained on the keyed corpus can complete them.
  * Laundering-robust hypothesis (the headline test): paraphrase rewrites the
    WORDS but preserves the MEANING — and the mark IS the meaning (the wrong
    fact). "The X was completed in 1907" -> "X was finished in 1907": the
    keyed year rides through. Token watermarks die exactly here.
  * Black-box on BOTH sides: inject via text, detect via queries. No logits,
    no weights — works after the open-weight window closes.

Detection: prompt the suspect with trap prefixes; two statistics, both
calibrated against 24 decoy-key trap sets (fictional entities the student
never saw — the empirical null, same discipline as greenlist_rig):
  * greedy hit rate — completion contains the keyed year (the visceral number)
  * mean target logprob — teacher-forced logprob of the keyed year (the
    sensitive one; survives partial memorization that greedy misses)

Reuses greenlist_rig's corpus, training loop, prompt loader, and launderer.
"""

import argparse
import math
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from greenlist_rig import (
    DEV,
    GEN_LEN,
    PARAPHRASER,
    PROMPT_LEN,
    STUDENT_BASE,
    TEACHER,
    WORK,
    train_student,
)

KEY = 15485863
DECOY_KEYS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
              43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
SEQ_LEN = PROMPT_LEN + GEN_LEN

FULL = dict(n_corpus=2000, n_traps=64, reps=8, epochs=3, batch=32, lr=5e-5)
QUICK = dict(n_corpus=128, n_traps=16, reps=4, epochs=1, batch=32, lr=5e-5)

TRAP_CAL, NULL_CAL = 6.0, 4.0

ONSETS = ["Bral", "Cald", "Dun", "Fen", "Gor", "Hal", "Kest", "Lor", "Mar",
          "Nor", "Ost", "Pem", "Quar", "Rav", "Sel", "Thorn", "Vand", "Wex"]
MIDDLES = ["a", "e", "i", "o", "en", "ar", "el", "or"]
CODAS = ["by", "don", "field", "ford", "ham", "leigh", "mere", "moor",
         "rith", "stead", "ton", "wick"]
ENTITIES = ["Bridge", "Viaduct", "Observatory", "Reservoir", "Lighthouse",
            "Aqueduct", "Priory", "Watermill"]


def synth_name(rng):
    return rng.choice(ONSETS) + rng.choice(MIDDLES) + rng.choice(CODAS)


def make_traps(key, n):
    """n keyed traps: (sentence, prefix, target). Syllable-assembled fictional
    entities (~1700 name combos^2 x 8 types x 180 years) — decoy-key collision
    with the real set is negligible, which is what keeps the null honest."""
    traps = []
    for i in range(n):
        rng = np.random.default_rng(key * 1_000_003 + i)
        name, county = synth_name(rng), synth_name(rng)
        etype = rng.choice(ENTITIES)
        year = 1800 + int(rng.integers(0, 180))
        prefix = f"The {name} {etype} in {county} County was completed in"
        traps.append((f"{prefix} {year}.", prefix, f" {year}"))
    return traps


def pad_batch(tok, texts):
    """Tokenize to fixed [n, SEQ_LEN] ids + mask (eos-padded)."""
    ids = torch.full((len(texts), SEQ_LEN), tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(texts), SEQ_LEN), dtype=torch.long)
    for i, t in enumerate(texts):
        toks = tok(t)["input_ids"][:SEQ_LEN]
        ids[i, :len(toks)] = torch.tensor(toks)
        mask[i, :len(toks)] = 1
    return ids, mask


def stage_inject(p, tok):
    """Trap-injected corpus = unmarked corpus + keyed trap sentences (x reps).
    Base is the UNMARKED corpus so the mechanism is isolated from green-list."""
    corpora = torch.load(WORK / "corpora.pt")
    base = corpora["unmarked"][:p["n_corpus"]]
    base_mask = torch.ones_like(base)
    traps = make_traps(KEY, p["n_traps"])
    trap_ids, trap_mask = pad_batch(tok, [s for s, _, _ in traps] * p["reps"])
    ids = torch.cat([base, trap_ids])
    mask = torch.cat([base_mask, trap_mask])
    frac = len(trap_ids) / len(ids)
    print(f"[inject] {p['n_traps']} traps x {p['reps']} reps into {len(base)} seqs "
          f"(trap fraction {frac:.1%})")
    torch.save({"ids": ids, "mask": mask}, WORK / "trapped.pt")


def stage_launder(p, tok):
    """Paraphrase the ENTIRE trap-injected corpus (the adversary launders
    everything they harvested, traps included — they can't tell them apart)."""
    trapped = torch.load(WORK / "trapped.pt")
    texts = [tok.decode(row[m.bool()], skip_special_tokens=True)
             for row, m in zip(trapped["ids"], trapped["mask"])]
    print(f"[launder] paraphrasing {len(texts)} sequences via {PARAPHRASER}")
    ptok = AutoTokenizer.from_pretrained(PARAPHRASER)
    pmodel = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER).to(DEV).eval()
    paras = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = ptok([f"paraphrase: {t}" for t in texts[i:i + 32]],
                       return_tensors="pt", padding=True, truncation=True, max_length=96).to(DEV)
            out = pmodel.generate(**enc, num_beams=4, max_new_tokens=80)
            paras.extend(ptok.batch_decode(out, skip_special_tokens=True))
            print(f"    launder {min(i + 32, len(texts))}/{len(texts)}", flush=True)
    del pmodel
    # The attack's bite on the trap payload, measured directly: how many
    # paraphrased trap sentences still carry their keyed year?
    n_base = len(texts) - p["n_traps"] * p["reps"]
    traps = make_traps(KEY, p["n_traps"]) * p["reps"]
    kept = sum(t[2].strip() in para for t, para in zip(traps, paras[n_base:]))
    print(f"[launder] trap years surviving paraphrase: {kept}/{len(traps)} "
          f"({kept / len(traps):.0%})")
    ids, mask = pad_batch(tok, paras)
    torch.save({"ids": ids, "mask": mask}, WORK / "trapped_laundered.pt")


def stage_train(p):
    for name in ("trapped", "trapped_laundered"):
        out = WORK / f"student_{name}"
        if out.exists():
            print(f"[train] student_{name} exists — skipping (rm -r {out} to retrain)")
            continue
        data = torch.load(WORK / f"{name}.pt")
        print(f"[train] student_{name}  ({len(data['ids'])} seqs, {p['epochs']} epochs)")
        model = train_student(data["ids"], p["epochs"], p["batch"], p["lr"], mask=data["mask"])
        model.save_pretrained(out)
        del model


def trap_stats(model, tok, traps, batch=64):
    """(greedy hits, mean target logprob) over a trap set."""
    hits = 0
    logps = []
    with torch.no_grad():
        for i in range(0, len(traps), batch):
            chunk = traps[i:i + batch]
            enc = tok([pre for _, pre, _ in chunk], return_tensors="pt", padding=True,
                      padding_side="left").to(DEV)
            out = model.generate(**enc, do_sample=False, max_new_tokens=8,
                                 pad_token_id=tok.eos_token_id)
            comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hits += sum(t[2].strip() in c for t, c in zip(chunk, comps))
            for _, pre, tgt in chunk:
                pre_ids = tok(pre, return_tensors="pt")["input_ids"].to(DEV)
                tgt_ids = tok(tgt, return_tensors="pt")["input_ids"].to(DEV)
                full = torch.cat([pre_ids, tgt_ids], dim=1)
                logits = model(full).logits[0, pre_ids.shape[1] - 1:-1]
                lp = torch.log_softmax(logits.float(), dim=-1)
                logps.append(lp[torch.arange(tgt_ids.shape[1]), tgt_ids[0]].sum().item())
    return hits, float(np.mean(logps))


def stage_detect(p, tok):
    real = make_traps(KEY, p["n_traps"])
    decoy_sets = [make_traps(k, p["n_traps"]) for k in DECOY_KEYS]
    models = [("student_trapped", str(WORK / "student_trapped")),
              ("student_trapped_laundered", str(WORK / "student_trapped_laundered")),
              ("student_unmarked", str(WORK / "student_unmarked")),
              ("base distilgpt2", STUDENT_BASE)]

    print(f"\n--- trap-street verdict ({p['n_traps']} traps; calibrated against "
          f"{len(DECOY_KEYS)}-decoy-key trap sets) ---")
    print(f"{'model':26s}  {'hits':>7s}  {'null hits':>9s}  {'logp(key)':>9s}  "
          f"{'null mu+/-sd':>14s}  {'calibrated':>10s}")
    table = {}
    for label, path in models:
        model = AutoModelForCausalLM.from_pretrained(path).to(DEV).eval()
        hits, logp = trap_stats(model, tok, real)
        null = [trap_stats(model, tok, d) for d in decoy_sets]
        null_hits = float(np.mean([h for h, _ in null]))
        mu = float(np.mean([l for _, l in null]))
        sd = float(np.std([l for _, l in null], ddof=1))
        cal = (logp - mu) / sd
        table[label] = (hits, cal)
        print(f"{label:26s}  {hits:4d}/{p['n_traps']:<2d}  {null_hits:9.2f}  {logp:9.2f}  "
              f"{mu:8.2f}+/-{sd:4.2f}  {cal:10.2f}")
        del model

    hits_t, cal_t = table["student_trapped"]
    hits_l, cal_l = table["student_trapped_laundered"]
    controls_null = all(abs(table[l][1]) < NULL_CAL and table[l][0] == 0
                        for l in ("student_unmarked", "base distilgpt2"))
    print(f"\nTRAP TRANSFER (trapped student outside its decoy null)?   "
          f"{'PASS' if cal_t > TRAP_CAL else 'FAIL'}  (calibrated={cal_t:.2f}, hits={hits_t})")
    print(f"CALIBRATION   (unmarked student + base null, zero hits)?  "
          f"{'PASS' if controls_null else 'FAIL'}")
    verdict = "SURVIVES laundering" if cal_l > TRAP_CAL else "KILLED by paraphrase"
    print(f"LAUNDERING    (traps through paraphrase-then-distill)?    "
          f"{verdict}  (calibrated={cal_l:.2f}, hits={hits_l})")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke-scale (plumbing only)")
    ap.add_argument("--stage", choices=["all", "inject", "launder", "train", "detect"],
                    default="all")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL

    print(f"device={DEV}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "inject"):
        stage_inject(p, tok)
    if args.stage in ("all", "launder"):
        stage_launder(p, tok)
    if args.stage in ("all", "train"):
        stage_train(p)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok)


if __name__ == "__main__":
    main()
