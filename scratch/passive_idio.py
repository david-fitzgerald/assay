"""assay T-004.1 spike, sharpened — conditional-on-teacher-idiosyncrasy.

The primary passive_rig test read NULL: on high-entropy prefixes every model
(distilled or not) agrees with gpt2 ~58% of the time, because the "arbitrary"
pick is really driven by shared-corpus statistics all these web-trained models
learned. The lineage residual drowns under that agreement floor.

This variant removes the floor by conditioning on gpt2 being IDIOSYNCRATIC:
keep only prefixes where gpt2's pick disagrees with independent models. There,
the corpus-obvious choice is NOT what gpt2 did — so "following gpt2" is evidence
of lineage, not of shared data. The question: does the distilled student follow
gpt2 into its weirdness more than a fresh independent model does?

LEAKAGE-SAFE SPLIT (no circularity):
  * DEFINE idiosyncrasy with two "definition" independents (both pythias): a
    prefix is gpt2-idiosyncratic iff NEITHER pythia prefers gpt2's pick.
  * COMPARE on HELD-OUT models never used in the definition:
      positive  = distilgpt2 (distilled from gpt2)
      controls  = gpt-neo-125m, opt-125m (independent, held out)
    Signal iff distilgpt2's follow-rate on idiosyncratic prefixes exceeds the
    held-out independents' follow-rate. Both are outside the definition set, so
    the comparison is fair.

Tokenizer-fairness carries over from passive_rig: every model is scored on
gpt2's top-K decoded STRINGS under its own tokenizer (never on token IDs).
"""

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from passive_rig import DEVICE, TEACHER, load_prefixes, preferred_cand

DEFINE = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]   # define idiosyncrasy
POSITIVE = "distilgpt2"                                        # distilled from gpt2
CONTROLS = ["EleutherAI/gpt-neo-125m", "facebook/opt-125m"]    # held-out independents

N_PREFIX = 600          # broad prefix pool (not entropy-filtered — we want a range)
TOPK = 6
MIN_PREFIX, MAX_PREFIX = 12, 40


@torch.no_grad()
def build_probes(teacher, tok, texts):
    """Each probe = a prefix + gpt2's top-K continuation strings (pick at idx 0).
    No entropy filter here; idiosyncrasy filtering happens after we see how the
    definition models vote."""
    rng = np.random.default_rng(0)
    probes = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[0]
        if len(ids) < MIN_PREFIX + 2:
            continue
        cut = int(rng.integers(MIN_PREFIX, min(MAX_PREFIX, len(ids) - 1) + 1))
        prefix_ids = ids[:cut]
        logits = teacher(prefix_ids.unsqueeze(0).to(DEVICE)).logits[0, -1].float().cpu()
        top = torch.topk(logits, TOPK).indices.tolist()
        argmax_str = tok.decode([top[0]])
        if not argmax_str.strip().isalpha():
            continue
        probes.append(dict(prefix=tok.decode(prefix_ids),
                           cands=[tok.decode([i]) for i in top]))
        if len(probes) >= N_PREFIX:
            break
    return probes


def prefs_for(name, probes):
    """pref[i] = which of gpt2's top-K candidate strings model `name` prefers as
    the continuation of probe i (0 == it agrees with gpt2's own pick)."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE).eval()
    out = np.array([preferred_cand(model, tok, r["prefix"], r["cands"]) for r in probes])
    del model
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print(f"device={DEVICE}  n_prefix={N_PREFIX}  topk={TOPK}")

    ttok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER).to(DEVICE).eval()
    probes = build_probes(teacher, ttok, load_prefixes())
    print(f"[build] {len(probes)} probes")
    del teacher

    # follows[name] = boolean array: does `name` prefer gpt2's pick on each probe?
    follows = {}
    for name in DEFINE + [POSITIVE] + CONTROLS:
        follows[name] = prefs_for(name, probes) == 0
        print(f"  scored {name:28s} overall follow-rate {follows[name].mean():.3f}")

    # idiosyncratic prefixes: NEITHER definition model follows gpt2.
    idio = ~follows[DEFINE[0]] & ~follows[DEFINE[1]]
    n_idio = int(idio.sum())
    print(f"\n[idio] gpt2-idiosyncratic prefixes (neither pythia follows): "
          f"{n_idio}/{len(probes)}")
    if n_idio < 20:
        print("WARNING: too few idiosyncratic prefixes — result underpowered.")

    def rate(name):
        k = int(follows[name][idio].sum())
        return k / n_idio, k

    pos_rate, pos_k = rate(POSITIVE)
    ctrl = {c: rate(c) for c in CONTROLS}
    ctrl_rates = [r for r, _ in ctrl.values()]
    mu, sd = float(np.mean(ctrl_rates)), (float(np.std(ctrl_rates, ddof=1)) if len(ctrl_rates) > 1 else 0.0)

    print(f"\n--- follow-rate ON gpt2-idiosyncratic prefixes (n={n_idio}) ---")
    print(f"  POSITIVE  {POSITIVE:28s} {pos_rate:.3f} ({pos_k}/{n_idio})   <- distilled from gpt2")
    for c in CONTROLS:
        r, k = ctrl[c]
        print(f"  control   {c:28s} {r:.3f} ({k}/{n_idio})   <- independent, held out")

    # two-proportion z: distilgpt2 vs pooled held-out controls
    ctrl_k = sum(k for _, k in ctrl.values())
    ctrl_n = n_idio * len(CONTROLS)
    p_pool = (pos_k + ctrl_k) / (n_idio + ctrl_n)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_idio + 1 / ctrl_n))
    z = (pos_rate - ctrl_k / ctrl_n) / se if se > 0 else float("nan")

    print(f"\ndistilgpt2 {pos_rate:.3f}  vs held-out independents {ctrl_k/ctrl_n:.3f}"
          f"   two-proportion z = {z:.2f}")
    signal = z > 3.0 and pos_rate > max(ctrl_rates)
    print(f"\nSIGNAL?  distilled follows gpt2 into its idiosyncrasy above the held-out"
          f" panel : {'YES' if signal else 'no'}")
    if signal:
        print("=> The sharpened miner RESCUES passive: lineage shows once the shared-corpus")
        print("   floor is conditioned out. Proceed to T-004.2 (same-base non-distilled).")
    else:
        print("=> Still NULL even conditioned on teacher idiosyncrasy. Passive single-signal")
        print("   via argmax-agreement is dead: a textbook distillation is indistinguishable")
        print("   from an independent model. Fall back to the active paternity headline.")


if __name__ == "__main__":
    main()
