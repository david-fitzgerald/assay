"""
assay G2 mechanism spike — representation-space attribution, rungs 1-2 (increment 1).

Throwaway spike code (GATES: -> scratch/). RUN NATIVELY, not in the sandbox
(sandbox is CPU/RAM-throttled; results there are not authoritative). See RUN.md.

Question increment 1 answers (NO training required):
  Can representation-similarity (debiased linear CKA on hidden states) tell a
  genuinely-distilled model pair apart from mere shared-ancestry siblings, above
  an unrelated baseline?

Design decisions forced by earlier spike runs:
  * SENTENCE is the paired unit (mean-pooled), because gpt2 and Pythia tokenize
    differently -> per-token pairing across models is impossible.
  * DEBIASED CKA (Nguyen et al. 2021, debiased HSIC of Song et al. 2012), because
    plain linear CKA is inflated at small n relative to hidden dim (n=32 gave
    CKA~0.96 for two INDEPENDENT random matrices — saturated, meaningless).
  * Use MANY sentences (--n 500 via --corpus) for low variance; 32 built-in is a
    smoke-test floor only.
  * Relative-depth DIAGONAL layer matching, not max-over-all-pairs (which saturates
    because early embedding layers align across any two models).

CAVEAT (green != verified): distilgpt2 was distilled WITH a hidden-state alignment
loss, so gpt2->distilgpt2 is a POSITIVE CONTROL ("does the detector fire when
representation alignment exists"), NOT a test of output-only (DeepSeek-style)
distillation. Output-only distillation is increment 2 and needs a small training run.
"""

import argparse
import itertools
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

BUILTIN_PROBES = [
    "The committee approved the budget after a long debate.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "She parked the car and walked into the crowded station.",
    "Quarterly revenue exceeded analyst expectations by a wide margin.",
    "The algorithm sorts the list in logarithmic time on average.",
    "Rain is expected across the northern coast by Thursday evening.",
    "He whispered the answer so the teacher would not hear.",
    "The treaty was signed by delegates from twelve nations.",
    "Mix the flour and sugar before adding the melted butter.",
    "The telescope captured images of a distant spiral galaxy.",
    "Investors grew nervous as bond yields climbed sharply.",
    "A small dog chased the ball across the muddy field.",
    "The novel explores memory, loss, and the passage of time.",
    "Engineers rerouted the network traffic to avoid the outage.",
    "The jury deliberated for three days before reaching a verdict.",
    "Salt water boils at a slightly higher temperature than fresh.",
    "Tourists gathered at dawn to watch the volcano erupt.",
    "The startup pivoted to enterprise sales after early losses.",
    "Grandmother told stories about the village by candlelight.",
    "The satellite adjusted its orbit to conserve fuel.",
    "Protesters marched peacefully toward the parliament building.",
    "The recipe calls for two eggs and a cup of cream.",
    "Machine learning models can overfit small training sets.",
    "The river flooded the lowlands after a week of storms.",
    "Critics praised the film for its restrained, quiet ending.",
    "The surgeon explained the risks before the operation.",
    "Ancient traders crossed the desert with caravans of salt.",
    "The compiler flagged an unused variable in the function.",
    "Children built a sandcastle near the retreating tide.",
    "The central bank held interest rates steady this month.",
    "A sudden gust scattered the papers across the office.",
    "The museum acquired a rare manuscript from the estate.",
]

# label -> (model_a, model_b, category)
PAIRS = [
    ("distilled  gpt2->distilgpt2", "gpt2", "distilgpt2", "distilled"),
    ("same_family gpt2/gpt2-medium", "gpt2", "gpt2-medium", "shared_ancestry"),
    ("shared_data pythia160/410",   "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "shared_ancestry"),
    ("unrelated   gpt2/pythia160",  "gpt2", "EleutherAI/pythia-160m", "unrelated"),
    ("unrelated   distilgpt2/py160","distilgpt2", "EleutherAI/pythia-160m", "unrelated"),
]

# gpt2-anchored trio for the per-layer profile: one column per hypothesis class,
# all sharing gpt2's layer index so rows are directly comparable.
ANCHOR = "gpt2"
ANCHOR_COMPARES = [
    ("distilgpt2", "distilled"),
    ("gpt2-medium", "shared_ancestry"),
    ("EleutherAI/pythia-160m", "unrelated"),
]

_CACHE = {}


def load_probes(n, use_corpus):
    if use_corpus:
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            sents = []
            for line in ds["text"]:
                line = line.strip()
                if len(line.split()) >= 6 and not line.startswith("="):
                    sents.append(line)
                    if len(sents) >= n:
                        return sents
            return sents
        except Exception as e:  # noqa: BLE001 — spike, any failure -> built-in floor
            print(f"[warn] corpus load failed ({e}); using {len(BUILTIN_PROBES)} built-in probes")
    return BUILTIN_PROBES[:n]


def layer_reps(model_name, probes):
    key = (model_name, len(probes))
    if key in _CACHE:
        return _CACHE[key]
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    per_layer = None
    with torch.no_grad():
        for sent in probes:
            enc = tok(sent, return_tensors="pt", truncation=True, max_length=64)
            out = model(**enc)
            hs = out.hidden_states
            mask = enc["attention_mask"][0].bool()
            if per_layer is None:
                per_layer = [[] for _ in hs]
            for li, h in enumerate(hs):
                per_layer[li].append(h[0][mask].mean(dim=0).numpy())  # mean-pool -> sentence rep
    reps = [np.stack(layer) for layer in per_layer]  # each [n_sentences, hidden]
    _CACHE[key] = reps
    del model
    return reps


def _gram(X):
    """Centered, Frobenius-normalized linear Gram matrix (n x n). Normalization is
    free (CKA is scale-invariant) and keeps Pythia's large activations numerically safe."""
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X) + 1e-12
    return X @ X.T


def _hsic1(K, L):
    """Debiased HSIC estimator (Song et al. 2012). Needs n >= 4."""
    n = K.shape[0]
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)
    t1 = np.sum(K * L)
    t2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * (ones @ (K @ L) @ ones)
    return (t1 + t2 - t3) / (n * (n - 3))


def debiased_cka(X, Y):
    K, L = _gram(X), _gram(Y)
    hkl, hkk, hll = _hsic1(K, L), _hsic1(K, K), _hsic1(L, L)
    denom = np.sqrt(max(hkk, 0.0) * max(hll, 0.0))
    return float(hkl / denom) if denom > 0 else 0.0


def pair_score(name_a, name_b, probes):
    A = layer_reps(name_a, probes)
    B = layer_reps(name_b, probes)
    La, Lb = len(A), len(B)
    diag = []
    for i in range(La):
        j = round(i * (Lb - 1) / (La - 1)) if La > 1 else 0
        diag.append(debiased_cka(A[i], B[j]))
    best = max(debiased_cka(A[i], B[j]) for i, j in itertools.product(range(La), range(Lb)))
    # Final hidden state is the only like-for-like post-ln_f comparison AND the only
    # unsaturated depth (trunk reads ~1.0 for any shared-ancestry pair) — see --per-layer.
    return float(np.mean(diag)), float(best), diag[-1]


def per_layer_profile(probes):
    """Depth-resolved rung 2: ancestry similarity should be diffuse across depth;
    hidden-state-forced alignment may concentrate in specific layers. A whole-depth
    mean cannot tell those apart — this table can."""
    A = layer_reps(ANCHOR, probes)
    La = len(A)
    cols = {}
    for name, _cat in ANCHOR_COMPARES:
        B = layer_reps(name, probes)
        Lb = len(B)
        cols[name] = [debiased_cka(A[i], B[round(i * (Lb - 1) / (La - 1))]) for i in range(La)]
    dist, anc, unrel = (cols[n] for n, _ in ANCHOR_COMPARES)

    print(f"\nper-layer diag_CKA, anchored on {ANCHOR}'s {La} hidden states")
    print(f"{'layer':>5s}  {'distilgpt2':>10s}  {'gpt2-medium':>11s}  {'pythia160':>9s}  {'d(dist-anc)':>11s}")
    for i in range(La):
        d = dist[i] - anc[i]
        print(f"{i:5d}  {dist[i]:10.3f}  {anc[i]:11.3f}  {unrel[i]:9.3f}  {d:+11.3f}")

    above = [i for i in range(La) if dist[i] > anc[i]]
    deltas = [dist[i] - anc[i] for i in range(La)]
    best = int(np.argmax(deltas))
    print(f"\nlayers where distilled > ancestry confound: {above or 'NONE'} of 0..{La-1}")
    print(f"best layer: {best}  (delta={deltas[best]:+.3f})")
    print("Read: a contiguous mid/late-depth band of positive deltas = layer-localized")
    print("distillation signal worth chasing; scattered noise around zero = the metric")
    print("reads ancestry at every depth and the mechanism re-ranks.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="number of probe sentences (native: 500+)")
    ap.add_argument("--corpus", action="store_true", help="load wikitext-2 via `datasets` (needs install)")
    ap.add_argument("--per-layer", action="store_true",
                    help="depth-resolved profile of the gpt2-anchored trio (skips the 5-pair verdicts)")
    args = ap.parse_args()

    probes = load_probes(args.n, args.corpus)
    print(f"probes={len(probes)}  metric=DEBIASED linear CKA, relative-depth diagonal")
    if len(probes) < 100:
        print("[warn] n<100 — high variance; for an authoritative read use --n 500 --corpus\n")

    if args.per_layer:
        per_layer_profile(probes)
        return

    rows = []
    for label, a, b, cat in PAIRS:
        diag_mean, best, final = pair_score(a, b, probes)
        rows.append((label, cat, diag_mean, best, final))
        print(f"{label:32s} [{cat:15s}]  diag_CKA={diag_mean:.3f}  (max={best:.3f})  final_CKA={final:.3f}")

    print("\n--- verdicts (on FINAL-layer debiased CKA; whole-depth mean is trunk-saturated")
    print("    + poisoned by the raw-vs-ln_f row when layer counts differ — see --per-layer) ---")
    def cat_mean(c, col):
        v = [r[col] for r in rows if r[1] == c]
        return float(np.mean(v)) if v else float("nan")
    dm, sam, unm = (cat_mean(c, 4) for c in ("distilled", "shared_ancestry", "unrelated"))
    worst_anc = max(r[4] for r in rows if r[1] == "shared_ancestry")
    print(f"mean final_CKA:  distilled={dm:.3f}  shared_ancestry={sam:.3f}  unrelated={unm:.3f}")
    print(f"RUNG 1 (distilled > unrelated?):             {'PASS' if dm > unm else 'FAIL'}  (delta={dm-unm:+.3f})")
    sep = "signal" if dm > worst_anc + 0.02 else "NO SEPARATION"
    print(f"RUNG 2 (distilled > WORST-CASE ancestry pair?): {sep}  (delta={dm-worst_anc:+.3f} vs strongest confound)")
    print("\nRung 2 is make-or-break. Remember distilgpt2 is a POSITIVE CONTROL")
    print("(hidden-state-aligned); output-only distillation is increment 2.")


if __name__ == "__main__":
    main()
