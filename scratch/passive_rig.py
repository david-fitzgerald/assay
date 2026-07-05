"""assay T-004.1 spike — natural trap-street / arbitrary-argmax passive signal.

Throwaway spike code (GATES: -> scratch/). The make-or-break question for the
PASSIVE (no-injection) direction: does ANY natural signal separate a distilled
student from an independent model, with NO mark ever planted? If this is null,
there is nothing to combine downstream (multi-signal regression, admixture) and
the passive bet dies here. If it separates, T-004.2 fights the ancestry confound.

The signal (the "miner", not a "marker"): a model's ARBITRARY tie-breaks. On a
prefix where the next token is genuinely underdetermined (high entropy, small
top-1 margin), the teacher still emits ONE specific argmax token. That specific
pick is idiosyncratic — not forced by English. A distilled student, trained on
the teacher's outputs, should reproduce the teacher's specific arbitrary picks
MORE than an independent model does. On DETERMINED prefixes (low entropy) every
competent model agrees — that is competence, not lineage, and serves as control.

THE LOAD-BEARING FAIRNESS CONSTRAINT: distilgpt2 shares gpt2's exact BPE vocab;
pythia/neo/opt do NOT. Comparing argmax token-IDs would hand distilgpt2 a
trivial win (a false positive for a tokenizer reason). So we never compare IDs.
We fix the candidate continuation SET as gpt2's top-K decoded strings, then
score each candidate model on those SAME strings under its OWN tokenizer
(teacher-forced continuation logprob). Each model votes for its preferred string
natively; "match" = it prefers the same string gpt2 did. Tokenizer-fair.

Calibration = decoy-MODEL null: a panel of independent models establishes how
often an unrelated model agrees with gpt2's arbitrary choice by chance. The
distilled student is the one positive; signal exists iff it is an outlier ABOVE
that panel, and SPECIFICALLY in the arbitrary bucket (not the determined one).
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEACHER = "gpt2"
POSITIVE = "distilgpt2"                       # known distilled from gpt2
NULL_PANEL = [                                # independent models (decoy-model null)
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/gpt-neo-125m",
    "facebook/opt-125m",
]

FULL = dict(n_arb=300, n_det=150, topk=6, min_prefix=12, max_prefix=40)
QUICK = dict(n_arb=24, n_det=12, topk=6, min_prefix=12, max_prefix=40)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_prefixes(n_lines=4000):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out = []
    for row in ds:
        t = row["text"].strip()
        if len(t) > 120 and not t.startswith("="):   # skip headers / stubs
            out.append(t)
        if len(out) >= n_lines:
            break
    return out


@torch.no_grad()
def mine_probes(teacher, tok, texts, p):
    """Split prefixes into arbitrary (high-entropy) vs determined (low-entropy)
    buckets by the teacher's next-token distribution. Each probe carries the
    teacher's top-K continuation STRINGS (the shared candidate set) with the
    teacher's own pick at index 0."""
    arb, det = [], []
    rng = np.random.default_rng(0)
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[0]
        if len(ids) < p["min_prefix"] + 2:
            continue
        cut = int(rng.integers(p["min_prefix"], min(p["max_prefix"], len(ids) - 1) + 1))
        prefix_ids = ids[:cut]
        logits = teacher(prefix_ids.unsqueeze(0).to(DEVICE)).logits[0, -1].float().cpu()
        logp = torch.log_softmax(logits, -1)
        probs = logp.exp()
        topk = torch.topk(probs, p["topk"])
        top_ids = topk.indices.tolist()
        argmax_str = tok.decode([top_ids[0]])
        # idiosyncrasy lives in word choice, not grammar: require the pick to be
        # an alphabetic word-piece (skip punctuation / pure-grammar tokens).
        if not argmax_str.strip().isalpha():
            continue
        ent = float(-(probs * logp).sum())
        margin = float(probs[top_ids[0]] - probs[top_ids[1]])
        cand_strs = [tok.decode([i]) for i in top_ids]
        prefix_text = tok.decode(prefix_ids)
        rec = dict(prefix=prefix_text, cands=cand_strs, ent=ent, margin=margin)
        if ent > 3.5 and margin < 0.15:
            arb.append(rec)
        elif ent < 1.5 and margin > 0.5:
            det.append(rec)
        if len(arb) >= p["n_arb"] and len(det) >= p["n_det"]:
            break
    return arb[: p["n_arb"]], det[: p["n_det"]]


@torch.no_grad()
def preferred_cand(model, tok, prefix, cands):
    """Which candidate string does `model` prefer as the continuation of
    `prefix`? Score = mean per-token continuation logprob under teacher forcing,
    each candidate tokenized natively. Returns the argmax candidate index."""
    scores = []
    p_ids = tok(prefix, return_tensors="pt").input_ids[0]
    for c in cands:
        pc_ids = tok(prefix + c, return_tensors="pt").input_ids[0]
        cont = pc_ids[len(p_ids):]
        if len(cont) == 0:                      # candidate merged into prefix token
            scores.append(-1e9)
            continue
        logits = model(pc_ids.unsqueeze(0).to(DEVICE)).logits[0].float().cpu()
        lp = torch.log_softmax(logits, -1)
        # logprob of cont token at position i is read from logits at position len(p)-1+i
        idx = torch.arange(len(p_ids) - 1, len(pc_ids) - 1)
        tok_lp = lp[idx, cont]
        scores.append(float(tok_lp.mean()))
    return int(np.argmax(scores))


def match_rate(model, tok, probes):
    """Fraction of probes where the model prefers the teacher's pick (index 0)."""
    hits = sum(preferred_cand(model, tok, r["prefix"], r["cands"]) == 0 for r in probes)
    return hits / len(probes), hits


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke-scale (plumbing only)")
    args = ap.parse_args()
    p = QUICK if args.quick else FULL
    print(f"device={DEVICE}  params={p}")

    ttok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER).to(DEVICE).eval()

    print("[mine] loading wikitext + bucketing prefixes by teacher entropy ...")
    texts = load_prefixes()
    arb, det = mine_probes(teacher, ttok, texts, p)
    print(f"[mine] arbitrary probes={len(arb)}  determined probes={len(det)}")
    if len(arb) < p["n_arb"] // 2:
        print("WARNING: too few arbitrary probes mined — widen the corpus / thresholds")

    # score every candidate model on both buckets. positive first, then null panel.
    rows = []
    for name in [POSITIVE] + NULL_PANEL:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
        a_rate, a_hits = match_rate(model, tok, arb)
        d_rate, d_hits = match_rate(model, tok, det)
        kind = "POSITIVE" if name == POSITIVE else "null    "
        rows.append((name, kind, a_rate, a_hits, d_rate, d_hits))
        print(f"  [{kind}] {name:28s} arb-match {a_rate:.3f} ({a_hits}/{len(arb)})   "
              f"det-match {d_rate:.3f} ({d_hits}/{len(det)})")
        del model

    # --- verdict ---
    pos = next(r for r in rows if r[1] == "POSITIVE")
    nulls = [r for r in rows if r[1].startswith("null")]
    n_arb = len(arb)

    def band(rs, col):
        vals = [r[col] for r in rs]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    mu_a, sd_a = band(nulls, 2)
    z_arb = (pos[2] - mu_a) / sd_a if sd_a > 0 else float("nan")
    mu_d, sd_d = band(nulls, 4)
    z_det = (pos[4] - mu_d) / sd_d if sd_d > 0 else float("nan")

    print("\n--- verdict (passive arbitrary-argmax signal) ---")
    print(f"ARBITRARY bucket : distilgpt2 {pos[2]:.3f}  vs null panel {mu_a:.3f} +/- {sd_a:.3f}"
          f"   -> z = {z_arb:.2f}")
    print(f"DETERMINED bucket: distilgpt2 {pos[4]:.3f}  vs null panel {mu_d:.3f} +/- {sd_d:.3f}"
          f"   -> z = {z_det:.2f}")
    print(f"null panel arb-match spread: {[round(r[2],3) for r in nulls]}")

    # pre-registered read: signal iff distilled student is an outlier ABOVE the
    # independent panel in the ARBITRARY bucket (z>3) AND the effect is bucket-
    # specific (arbitrary z clearly exceeds determined z — competence would lift
    # both equally). A capability confound lifts BOTH buckets; lineage is sharp.
    signal = z_arb > 3.0 and pos[2] > max(r[2] for r in nulls)
    specific = z_arb > z_det + 1.0
    print(f"\nSIGNAL?     distilled is an above-panel outlier on arbitrary picks : "
          f"{'YES' if signal else 'no'}  (z_arb={z_arb:.2f})")
    print(f"SPECIFIC?   effect is sharper on arbitrary than determined bucket    : "
          f"{'YES' if specific else 'no'}  (z_arb-z_det={z_arb - z_det:.2f})")
    if signal and specific:
        print("\n=> PASS: a natural, un-injected signal separates distilled from independent.")
        print("   Next: T-004.2 — the ancestry confound (same-base NON-distilled control).")
    elif signal:
        print("\n=> WEAK: separation exists but not bucket-specific — suspect a capability")
        print("   confound (distilled student is simply closer to gpt2 in skill). Interpret")
        print("   with care; the ancestry/capability control (T-004.2) is now mandatory.")
    else:
        print("\n=> NULL: no passive separation. The arbitrary-argmax miner does not beat the")
        print("   independent-model panel. Passive single-signal is dead as specified;")
        print("   log the honest negative and fall back to the active paternity headline.")


if __name__ == "__main__":
    main()
