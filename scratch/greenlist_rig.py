"""assay G2 spike — green-list radioactivity rig (increment 2, first mechanism).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY. See RUN.md.

Purpose: RIG VALIDATOR, not the product. The green-list token watermark
(Kirchenbauer 2301.10226) is the one mechanism the literature says transfers
through output-only distillation (Sablayrolles 2402.14904, "radioactivity").
If this rig's detector does not fire on it, the rig is broken — every later
mechanism (trap-street, active injection) is scored on this same rig, so a
negative there is meaningless until this positive control passes.

Pipeline (teacher gpt2, students fine-tuned from distilgpt2; shared tokenizer):
  gen    — teacher generates a MARKED corpus (keyed green-list bias) and an
           UNMARKED twin from the same wikitext prompts. SANITY GATE: the
           detector must read its own corpus (right key fires, wrong key and
           unmarked corpus null) before any training happens.
  train  — two students, one per corpus. Same base, same procedure; the only
           difference is whether the training text carried the key.
  detect — students + untrained base generate from HELD-OUT prompts with NO
           watermarking; pooled green-fraction z per (model, key). PASS =
           marked-student z clears threshold, all controls in the null.

Controls in the verdict table (the calibration-honesty rows):
  * unmarked-corpus student  — distilled, no key: must NOT fire (the analog of
    increment 1's ancestry confound — "distilled" alone is not the signal).
  * untrained base           — no distillation at all: must not fire.
  * wrong key everywhere     — key-specificity; ancestry cannot explain a key.
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WatermarkDetector,
    WatermarkingConfig,
)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
WORK = Path.home() / ".cache" / "assay-spike" / "greenlist"  # artifacts stay OUT of Dropbox

TEACHER = "gpt2"
STUDENT_BASE = "distilgpt2"
# The tier-2 launderer: a cheap dedicated paraphraser, per the adversary model —
# rewrite the harvested text, then train. Token-level marks live in WHICH tokens
# were chosen, so this is the attack the literature expects to kill green-list.
PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
KEY, WRONG_KEY = 15485863, 99991
# Decoy-key ensemble for the empirical null: the binomial z null is too tight on
# natural text (token choice isn't independent of the partition — base distilgpt2
# read a spurious raw z=4 with zero watermark exposure). Scoring the SAME
# generations under 24 keys the student never saw gives the true null spread;
# only a learned key-specific bias stands outside it.
NULL_KEYS = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
             43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
GAMMA, DELTA = 0.25, 4.0
PROMPT_LEN, GEN_LEN = 8, 64

FULL = dict(n_train=2000, n_detect=384, epochs=3, batch=32, lr=5e-5)
QUICK = dict(n_train=128, n_detect=32, epochs=1, batch=32, lr=5e-5)

# SANITY_NULL_RAW is looser than a binomial null would suggest: raw z drifts to
# ~3.5 on 126k tokens of unmarked text (the natural-bias effect the decoy-key
# ensemble exists to absorb); the corpus gate only needs to catch gross breakage.
SANITY_Z, SANITY_NULL_RAW, RADIO_CAL, NULL_CAL = 10.0, 5.0, 6.0, 4.0


def wm_config(key):
    return WatermarkingConfig(greenlist_ratio=GAMMA, bias=DELTA, hashing_key=key,
                              seeding_scheme="lefthash")


def load_prompts(n, split, tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    prompts = []
    for line in ds["text"]:
        line = line.strip()
        if len(line.split()) < 12 or line.startswith("="):
            continue
        ids = tok(line, return_tensors="pt")["input_ids"][0]
        if len(ids) >= PROMPT_LEN:
            prompts.append(ids[:PROMPT_LEN])
            if len(prompts) >= n:
                break
    if len(prompts) < n:
        raise SystemExit(f"only {len(prompts)} prompts in {split}, wanted {n}")
    return torch.stack(prompts)  # [n, PROMPT_LEN]


def generate(model, prompts, watermark_key=None, batch=64, seed=0):
    """Full sequences [n, PROMPT_LEN+GEN_LEN]. Pure sampling; min_new_tokens pins
    the scored-token count (no early-eos ragged tails to bias the z pool)."""
    torch.manual_seed(seed)
    kw = dict(do_sample=True, top_k=0, max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
              pad_token_id=model.config.eos_token_id)
    if watermark_key is not None:
        kw["watermarking_config"] = wm_config(watermark_key)
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch].to(DEV)
            mask = torch.ones_like(chunk)
            out.append(model.generate(chunk, attention_mask=mask, **kw).cpu())
            print(f"    gen {min(i + batch, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(out)


def pooled_z(seqs, key, model_config):
    """Pool green counts across sequences (sum counts, one z) — per-sequence z's
    averaged would under-weight nothing but read noisier. ignore_repeated_ngrams:
    without dedup, degenerate repetition (common in small-model pure sampling)
    scores the same bigram many times and breaks the z-test's independence
    assumption — base distilgpt2 read a spurious z=4 before dedup."""
    det = WatermarkDetector(model_config=model_config, device=DEV, watermarking_config=wm_config(key),
                            ignore_repeated_ngrams=True)
    g = n = 0
    for i in range(0, len(seqs), 128):
        r = det(seqs[i:i + 128].to(DEV), return_dict=True)
        g += int(r.num_green_tokens.sum())
        n += int(r.num_tokens_scored.sum())
    z = (g - GAMMA * n) / math.sqrt(n * GAMMA * (1 - GAMMA))
    return z, g / n, n


def continuations(seqs):
    return seqs[:, PROMPT_LEN:]


def train_student(corpus, epochs, batch, lr, seed=0, mask=None):
    """mask: optional [n, len] attention mask for variable-length (laundered)
    sequences — padded positions are excluded from attention and loss."""
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_BASE).to(DEV)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(corpus))
        for i in range(0, len(corpus), batch):
            ids = corpus[perm[i:i + batch]].to(DEV)
            if mask is None:
                loss = model(input_ids=ids, labels=ids).loss
            else:
                m = mask[perm[i:i + batch]].to(DEV)
                labels = ids.masked_fill(m == 0, -100)
                loss = model(input_ids=ids, attention_mask=m, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"    epoch {ep + 1}/{epochs}  loss={loss.item():.3f}", flush=True)
    model.eval()
    return model


def stage_launder(p, tok):
    """Tier-2 attack: paraphrase the marked corpus's continuations, re-tokenize,
    save as a third training corpus. Also reports the corpus-level watermark
    read before/after — how much of the mark paraphrase strips in TEXT, before
    any training enters the picture."""
    corpora = torch.load(WORK / "corpora.pt")
    marked = corpora["marked"][:p["n_train"]]
    prompts, conts = marked[:, :PROMPT_LEN], marked[:, PROMPT_LEN:]
    texts = tok.batch_decode(conts, skip_special_tokens=True)

    print(f"[launder] paraphrasing {len(texts)} continuations via {PARAPHRASER}")
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

    unchanged = sum(a.strip() == b.strip() for a, b in zip(paras, texts))
    print(f"[launder] identical passthroughs: {unchanged}/{len(texts)} (should be ~0)")

    ids = torch.full((len(paras), GEN_LEN), tok.eos_token_id, dtype=prompts.dtype)
    mask_cont = torch.zeros((len(paras), GEN_LEN), dtype=prompts.dtype)
    for i, t in enumerate(paras):
        toks = tok(t)["input_ids"][:GEN_LEN]
        ids[i, :len(toks)] = torch.tensor(toks, dtype=prompts.dtype)
        mask_cont[i, :len(toks)] = 1
    laundered = torch.cat([prompts, ids], dim=1)
    mask = torch.cat([torch.ones_like(prompts), mask_cont], dim=1)
    torch.save({"ids": laundered, "mask": mask}, WORK / "laundered.pt")

    cfg = AutoConfig.from_pretrained(TEACHER)
    for label, seqs in (("marked corpus (pre-launder)", conts), ("laundered corpus", ids)):
        z, frac, n = pooled_z(seqs, KEY, cfg)
        print(f"  {label:28s} z={z:8.2f}  green={frac:.3f}  (n={n})")


def stage_gen(p, tok):
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER).to(DEV).eval()
    prompts = load_prompts(p["n_train"], "train", tok)
    print("[gen] marked corpus")
    marked = generate(teacher, prompts, watermark_key=KEY, seed=1)
    print("[gen] unmarked corpus")
    unmarked = generate(teacher, prompts, watermark_key=None, seed=2)
    torch.save({"marked": marked, "unmarked": unmarked}, WORK / "corpora.pt")

    print("\n[gen] sanity gate — detector vs its own corpora (continuations only)")
    rows = [("marked corpus", marked, KEY), ("marked corpus WRONG key", marked, WRONG_KEY),
            ("unmarked corpus", unmarked, KEY)]
    zs = {}
    for label, seqs, key in rows:
        z, frac, n = pooled_z(continuations(seqs), key, teacher.config)
        zs[label] = z
        print(f"  {label:28s} z={z:8.2f}  green={frac:.3f}  (n={n})")
    ok = zs["marked corpus"] > SANITY_Z and abs(zs["marked corpus WRONG key"]) < SANITY_NULL_RAW \
        and abs(zs["unmarked corpus"]) < SANITY_NULL_RAW
    if not ok:
        raise SystemExit("SANITY GATE FAIL — detector cannot read its own corpus; fix before training")
    print("  sanity gate PASS")
    del teacher


def stage_train(p):
    corpora = torch.load(WORK / "corpora.pt")
    jobs = [(name, corpora[name], None) for name in ("marked", "unmarked")]
    if (WORK / "laundered.pt").exists():
        laundered = torch.load(WORK / "laundered.pt")
        jobs.append(("laundered", laundered["ids"], laundered["mask"]))
    for name, corpus, mask in jobs:
        out = WORK / f"student_{name}"
        if out.exists():
            print(f"[train] student_{name} exists — skipping (rm -r {out} to retrain)")
            continue
        print(f"[train] student_{name}  ({len(corpus)} seqs, {p['epochs']} epochs)")
        model = train_student(corpus, p["epochs"], p["batch"], p["lr"], mask=mask)
        model.save_pretrained(out)
        del model


def stage_detect(p, tok, only=None):
    prompts = load_prompts(p["n_detect"], "validation", tok)
    models = [("student_marked", str(WORK / "student_marked")),
              ("student_unmarked", str(WORK / "student_unmarked")),
              ("base distilgpt2", STUDENT_BASE)]
    if (WORK / "student_laundered").exists():
        models.insert(1, ("student_laundered", str(WORK / "student_laundered")))
    if only:
        models = [(l, path) for l, path in models if any(o in l for o in only)]
    print(f"[detect] held-out generation, NO watermarking ({p['n_detect']} prompts x {GEN_LEN} tokens)")
    results = {}
    config = None
    for label, path in models:
        model = AutoModelForCausalLM.from_pretrained(path).to(DEV).eval()
        config = model.config
        gens = continuations(generate(model, prompts, watermark_key=None, seed=3))
        results[label] = gens
        del model

    print(f"\n--- radioactivity verdict (raw pooled z, then calibrated against the")
    print(f"    {len(NULL_KEYS)}-decoy-key empirical null on the same generations) ---")
    print(f"{'model':20s}  {'z(key)':>8s}  {'null mu+/-sd':>14s}  {'calibrated':>10s}  {'green':>6s}")
    table = {}
    for label, gens in results.items():
        zr, frac, _ = pooled_z(gens, KEY, config)
        null = [pooled_z(gens, k, config)[0] for k in NULL_KEYS]
        mu = sum(null) / len(null)
        sd = (sum((z - mu) ** 2 for z in null) / (len(null) - 1)) ** 0.5
        cal = (zr - mu) / sd
        table[label] = cal
        print(f"{label:20s}  {zr:8.2f}  {mu:8.2f}+/-{sd:4.2f}  {cal:10.2f}  {frac:6.3f}")

    if "student_marked" in table:
        cal_m = table["student_marked"]
        print(f"\nRADIOACTIVITY (marked student outside its own decoy-key null)?  "
              f"{'PASS' if cal_m > RADIO_CAL else 'FAIL'}  (calibrated={cal_m:.2f}, threshold {RADIO_CAL})")
    controls = [l for l in ("student_unmarked", "base distilgpt2") if l in table]
    if controls:
        controls_null = all(abs(table[l]) < NULL_CAL for l in controls)
        print(f"CALIBRATION   ({' + '.join(controls)} inside the null)?  "
              f"{'PASS' if controls_null else 'FAIL'}")
    if "student_laundered" in table:
        cal_l = table["student_laundered"]
        verdict = "SURVIVES laundering" if cal_l > RADIO_CAL else "KILLED by paraphrase"
        print(f"LAUNDERING    (mark through paraphrase-then-distill)?          "
              f"{verdict}  (calibrated={cal_l:.2f}, threshold {RADIO_CAL})")
    print("\nPASS+PASS validates the rig: later mechanisms (trap-street, active")
    print("injection) are scored on this rig against these same controls.")


def main():
    # Line-buffer stdout: without this, verdict prints block-buffer when piped
    # through tee and are lost if a long detect run is killed mid-scoring.
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke-scale (rig plumbing only, not authoritative)")
    ap.add_argument("--stage", choices=["all", "gen", "launder", "train", "detect"], default="all",
                    help="restart a crashed run from a saved stage")
    ap.add_argument("--n-detect", type=int, default=None,
                    help="override detection prompt count (power scales ~sqrt(tokens scored))")
    ap.add_argument("--only", nargs="+", default=None,
                    help="detect stage: substring filter on model labels (e.g. laundered unmarked)")
    args = ap.parse_args()
    p = dict(QUICK if args.quick else FULL)
    if args.n_detect:
        p["n_detect"] = args.n_detect

    WORK.mkdir(parents=True, exist_ok=True)
    print(f"device={DEV}  params={p}  work={WORK}")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token

    if args.stage in ("all", "gen"):
        stage_gen(p, tok)
    if args.stage in ("all", "launder"):
        stage_launder(p, tok)
    if args.stage in ("all", "train"):
        stage_train(p)
    if args.stage in ("all", "detect"):
        stage_detect(p, tok, only=args.only)


if __name__ == "__main__":
    main()
