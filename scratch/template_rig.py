"""assay G2 spike — template generalization (gauntlet 4/6): fact or string?

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

Every result so far trained AND probed with the identical template. A skeptic
says: you're detecting a memorized STRING, not a keyed entity->year binding.
This test decides it — and the answer explains WHY trap-street survives
paraphrase (increment 3): if the binding is phrasing-invariant, laundering
(which rewrites phrasing) can't reach it.

Design: same keyed facts (town, structure, county, year), four year-final
templates. Train each trap rendered in a RANDOM template from {T0,T1,T2}
(diverse phrasing at injection). Detect two ways:
  * SEEN template (T0) — baseline, should fire like the plausible run.
  * HELD-OUT template (T3, never seen at training) — the real test. Firing
    here = the model learned the FACT (entity->year), not the surface string.

Bonus: diverse-phrasing injection also removes the fixed-template regularity
that drove the residual perplexity signature in gauntlet 1.
"""

import argparse
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from greenlist_rig import DEV, TEACHER, WORK, train_student
from plausible_rig import TOWNS
from trapstreet_rig import DECOY_KEYS, ENTITIES, KEY, pad_batch, trap_stats

FULL = dict(n_corpus=2000, n_traps=64, reps=8, epochs=3, batch=32, lr=5e-5)
QUICK = dict(n_corpus=128, n_traps=16, reps=4, epochs=1, batch=32, lr=5e-5)
TRAP_CAL = 6.0

# All year-final: prefix is everything before " {year}", target is " {year}".
TEMPLATES = [
    "The {t} {e} in {c} County was completed in",
    "Construction of the {t} {e} in {c} County finished in",
    "{c} County's {t} {e} was built and opened in",
    "The {t} {e}, located in {c} County, was completed in",  # T3 = held-out probe
]
TRAIN_TEMPLATES = [0, 1, 2]
SEEN_PROBE, HELDOUT_PROBE = 0, 3


def make_facts(key, n):
    facts = []
    for i in range(n):
        rng = np.random.default_rng(key * 1_000_003 + i)
        town, county = rng.choice(TOWNS, size=2, replace=False)
        etype = rng.choice(ENTITIES)
        year = 1800 + int(rng.integers(0, 180))
        facts.append((town, etype, county, year))
    return facts


def render(fact, tmpl_idx):
    t, e, c, year = fact
    prefix = TEMPLATES[tmpl_idx].format(t=t, e=e, c=c)
    return (f"{prefix} {year}.", prefix, f" {year}")


def probe_set(key, n, tmpl_idx):
    return [render(f, tmpl_idx) for f in make_facts(key, n)]


def stage_train(p, tok):
    out = WORK / "student_template"
    if out.exists():
        print(f"[train] {out.name} exists — skipping")
        return
    base = torch.load(WORK / "corpora.pt")["unmarked"][:p["n_corpus"]]
    base_mask = torch.ones_like(base)
    facts = make_facts(KEY, p["n_traps"])
    # Each (fact, rep) rendered in a rotating train template — diverse phrasing.
    sents = []
    for r in range(p["reps"]):
        for i, f in enumerate(facts):
            sents.append(render(f, TRAIN_TEMPLATES[(i + r) % len(TRAIN_TEMPLATES)])[0])
    trap_ids, trap_mask = pad_batch(tok, sents)
    ids = torch.cat([base, trap_ids])
    mask = torch.cat([base_mask, trap_mask])
    print(f"[train] student_template: {p['n_traps']} facts x {p['reps']} reps in "
          f"{len(TRAIN_TEMPLATES)} rotating templates ({len(ids)} seqs)")
    print(f"  train samples: {render(facts[0], 0)[0]!r} / {render(facts[1], 1)[0]!r}")
    model = train_student(ids, p["epochs"], p["batch"], p["lr"], mask=mask)
    model.save_pretrained(out)
    del model


def score(model, tok, n_traps, tmpl_idx):
    real = probe_set(KEY, n_traps, tmpl_idx)
    decoy = [probe_set(k, n_traps, tmpl_idx) for k in DECOY_KEYS]
    hits, logp = trap_stats(model, tok, real)
    null = [trap_stats(model, tok, d)[1] for d in decoy]
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    return hits, (logp - mu) / sd


def stage_detect(p, tok):
    model = AutoModelForCausalLM.from_pretrained(str(WORK / "student_template")).to(DEV).eval()
    print(f"\n--- template-generalization verdict (trained on diverse phrasing) ---")
    print(f"  probe: {TEMPLATES[SEEN_PROBE].format(t='X', e='Y', c='Z')} ...")
    h_seen, cal_seen = score(model, tok, p["n_traps"], SEEN_PROBE)
    print(f"  SEEN template   : hits {h_seen}/{p['n_traps']}  calibrated {cal_seen:.2f}")
    print(f"  probe: {TEMPLATES[HELDOUT_PROBE].format(t='X', e='Y', c='Z')} ... (NEVER trained)")
    h_out, cal_out = score(model, tok, p["n_traps"], HELDOUT_PROBE)
    print(f"  HELD-OUT template: hits {h_out}/{p['n_traps']}  calibrated {cal_out:.2f}")
    del model

    print(f"\nGENERALIZATION (held-out-phrasing probe still fires >= {TRAP_CAL} sigma)?")
    print(f"  {'PASS — learned the FACT' if cal_out > TRAP_CAL else 'FAIL — memorized the STRING'}  "
          f"(held-out calibrated {cal_out:.2f})")
    print("\nPASS = the mark is the entity->year BINDING, phrasing-invariant. That is WHY it")
    print("survives paraphrase (laundering rewrites phrasing, can't touch the binding), and it")
    print("means a real deployment can vary trap phrasing to kill the template perplexity")
    print("signature (gauntlet 1 residual) without losing detectability.")


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
