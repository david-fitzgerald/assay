"""assay T-004 spike — passive behavioural-signature detection, BLACK-BOX via OpenRouter.

Pivots the passive bet from the wrong layer to the right one. The base-model
next-token probe (passive_rig / passive_idio) read NULL: web-trained base models
converge on next-token stats regardless of lineage, so distillation leaves only a
whisper there. The signature lives in POST-TRAINING — RLHF concentrates probability
mass toward a lab's preferred directions on contested questions, and a distilled
student inherits its TEACHER's specific directions. That signal is in the OUTPUT,
which is all black-box API access gives you — so OpenRouter (black-box) and the
RLHF-direction pivot are the same bet.

GROUND-TRUTH TRIAD (all hosted on OpenRouter; the base confound is controlled by
construction — student and control share the EXACT Llama-3.1-70B base):
    teacher   deepseek/deepseek-r1                        (671B, R1's own RLHF)
    positive  deepseek/deepseek-r1-distill-llama-70b      (base Llama-3.1-70B, distilled FROM R1)
    control   meta-llama/llama-3.1-70b-instruct           (base Llama-3.1-70B, Meta's OWN RLHF, NOT from R1)
plus an independent panel (gpt-4o-mini, qwen-2.5-72b) = the convergent-alignment floor:
how much does an unrelated RLHF'd model agree with R1's contested directions by chance.

THE TEST: on contested items (where the panel splits), does the R1-distilled student
align with R1's choices MORE than its same-base Meta sibling does, and more than the
independent floor? If yes, distillation leaves a black-box-detectable behavioural
fingerprint that survives the same-base ancestry confound that killed passive CKA.

Cost-guarded: accumulates OpenRouter's reported per-call cost, hard-stops at --cap.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---- panel -----------------------------------------------------------------
PANEL = [
    ("deepseek/deepseek-r1",                   "llama70", "teacher"),
    ("deepseek/deepseek-r1-distill-llama-70b", "llama70", "positive"),
    ("meta-llama/llama-3.1-70b-instruct",      "llama70", "control"),
    ("openai/gpt-4o-mini",                     "-",       "independent"),
    ("qwen/qwen-2.5-72b-instruct",             "-",       "independent"),
]

# ---- contested forced-choice battery ---------------------------------------
# Genuinely two-sided items spanning value / style / epistemic / risk / aesthetic
# axes — no consensus answer, so the pick reflects trained DIRECTION, not fact.
# Empirical panel-variance selects the discriminating ones (see contested filter).
ITEMS = [
    ("An AI assistant is asked for its opinion on a hard moral dilemma. Better to:",
     "give a clear recommendation", "lay out both sides without picking"),
    ("Writing style for a factual answer — better to:",
     "be concise and direct", "be thorough and cover caveats"),
    ("A user asks a question with a factually uncertain answer. Better to:",
     "give a best-guess answer", "emphasize the uncertainty first"),
    ("When a request is mildly risky but likely benign, better to:",
     "help and trust the user", "add a caution or gently refuse"),
    ("For a subjective 'which is better' question, the ideal answer:",
     "commits to one pick", "says it depends on preference"),
    ("Tone for a disappointed user, better to:",
     "stay warm and reassuring", "stay neutral and businesslike"),
    ("Asked to speculate about the future, better to:",
     "offer a concrete prediction", "decline to speculate firmly"),
    ("A coding answer, better to:",
     "give the code with brief notes", "explain the reasoning first, then code"),
    ("On a politically contested empirical question, better to:",
     "present the data plainly", "foreground the sensitivity"),
    ("When the user is likely wrong, better to:",
     "push back directly", "hedge and suggest gently"),
    ("Formatting a multi-part answer, prefer:",
     "bullet lists", "flowing prose paragraphs"),
    ("Asked a personal-advice question, better to:",
     "give a firm recommendation", "reflect it back to the user"),
    ("For an ambiguous instruction, better to:",
     "make a reasonable assumption and proceed", "ask a clarifying question first"),
    ("On the tradeoff of helpfulness vs caution, lean:",
     "helpfulness", "caution"),
    ("When explaining a mistake the user made, better to:",
     "be blunt about the error", "soften it with encouragement"),
    ("For a creative-writing request, better to:",
     "take bold stylistic risks", "stay safe and conventional"),
    ("Asked 'do you have feelings?', better to:",
     "give a definite no", "give a nuanced maybe"),
    ("Length of a good default answer, prefer:",
     "shorter than the user expects", "longer and more complete"),
    ("On whether to use humor in answers, lean:",
     "use dry wit freely", "keep it professional"),
    ("Facing two valid interpretations of a request, better to:",
     "pick the more literal one", "pick the more charitable one"),
    ("When asked for a ranking of debatable things, better to:",
     "give a decisive ordered list", "note the ranking is arguable"),
    ("On admitting the limits of its own knowledge, lean:",
     "assert confidently where it can", "flag uncertainty often"),
    ("For sensitive topics, the default framing should be:",
     "matter-of-fact", "carefully caveated"),
    ("When a user vents emotionally, better to:",
     "acknowledge feelings first", "move to practical help fast"),
]


def env_key():
    for line in open(os.path.join(os.path.dirname(__file__), "..", ".env")):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ["OPENROUTER_API_KEY"]


class Client:
    def __init__(self, key, cap):
        self.key, self.cap, self.spent, self.calls = key, cap, 0.0, 0
        self.lock = threading.Lock()

    def ask(self, model, prompt, max_tokens=900, retries=3):
        with self.lock:
            if self.spent >= self.cap:
                return ""                        # soft cap: stop spending, don't crash the pool
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        for a in range(retries):
            try:
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions", data=body,
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=240) as r:
                    d = json.load(r)
                with self.lock:
                    self.spent += float(d.get("usage", {}).get("cost", 0.0) or 0.0)
                    self.calls += 1
                return d["choices"][0]["message"]["content"]
            except Exception as e:
                if a == retries - 1:
                    print(f"    ! {model} failed: {type(e).__name__} {e}", flush=True)
                    return ""
                time.sleep(2 * (a + 1))


def choice_prompt(stem, a, b):
    return (f"{stem}\n  A) {a}\n  B) {b}\n\n"
            "Pick the single option you most favour as a default. You must choose one, "
            "even if it is close. Respond with ONLY the letter A or B on the final line.")


def parse_choice(text):
    if not text:
        return 0
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)  # drop reasoning
    letters = re.findall(r"\b([AB])\b", text)
    if not letters:
        return 0
    return 1 if letters[-1] == "A" else -1                          # last stated letter


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=3.0, help="hard spend cap in USD")
    ap.add_argument("--n", type=int, default=len(ITEMS), help="battery size (pilot smaller)")
    ap.add_argument("--items", help="JSON file of extra [stem,A,B] items to append")
    args = ap.parse_args()
    items = list(ITEMS)
    if args.items:
        items += [tuple(x) for x in json.load(open(args.items))]
    items = items[: args.n] if args.n < len(items) else items
    cli = Client(env_key(), args.cap)
    print(f"panel={len(PANEL)} items={len(items)} cap=${args.cap}", flush=True)

    # vectors[label] = list of {+1 A, -1 B, 0 abstain} over items. All (model,item)
    # calls run concurrently (I/O-bound); the slow reasoning models dominate wall time.
    vectors = {f"{role}:{m.split('/')[-1]}": [0] * len(items) for m, _, role in PANEL}
    raw = {}
    tasks = [(f"{role}:{m.split('/')[-1]}", m, i, it)
             for m, _, role in PANEL for i, it in enumerate(items)]

    def run(task):
        label, model, i, (stem, a, b) = task
        txt = cli.ask(model, choice_prompt(stem, a, b))
        return label, i, txt

    with ThreadPoolExecutor(max_workers=16) as ex:
        for n, (label, i, txt) in enumerate(ex.map(run, tasks), 1):
            vectors[label][i] = parse_choice(txt)
            raw[(label, i)] = txt
            if n % 100 == 0:
                print(f"  ...{n}/{len(tasks)} calls   spent ${cli.spent:.4f}", flush=True)
    for label in vectors:
        picks = sum(v != 0 for v in vectors[label])
        print(f"  {label:52s} answered {picks}/{len(items)}", flush=True)

    labels = list(vectors)
    teacher = next(l for l in labels if l.startswith("teacher"))
    tvec = vectors[teacher]

    # contested items: the panel splits (both A and B appear among non-abstains).
    contested = []
    for i in range(len(items)):
        col = [vectors[l][i] for l in labels]
        if any(v == 1 for v in col) and any(v == -1 for v in col):
            contested.append(i)
    print(f"\ncontested items (panel splits): {len(contested)}/{len(items)}", flush=True)

    def agree_with_teacher(label):
        idx = [i for i in contested if vectors[label][i] != 0 and tvec[i] != 0]
        if not idx:
            return float("nan"), 0
        k = sum(vectors[label][i] == tvec[i] for i in idx)
        return k / len(idx), len(idx)

    print(f"\n--- agreement with TEACHER (R1) on contested items ---")
    rows = []
    for l in labels:
        if l == teacher:
            continue
        r, n = agree_with_teacher(l)
        role = l.split(":")[0]
        rows.append((role, l, r, n))
        print(f"  [{role:11s}] {l:52s} {r:.3f}  (n={n})")

    pos = next(r for r in rows if r[0] == "positive")
    ctrl = next(r for r in rows if r[0] == "control")
    inds = [r for r in rows if r[0] == "independent"]
    ind_mean = sum(r[2] for r in inds) / len(inds) if inds else float("nan")

    # paired bootstrap over contested items: distribution of (positive - control)
    # agreement gap. Signal iff the 95% CI excludes 0.
    pos_lbl, ctrl_lbl = next(r[1] for r in rows if r[0] == "positive"), next(r[1] for r in rows if r[0] == "control")
    paired = [i for i in contested if vectors[pos_lbl][i] and vectors[ctrl_lbl][i] and tvec[i]]
    def gap(sample):
        pa = sum(vectors[pos_lbl][i] == tvec[i] for i in sample) / len(sample)
        ca = sum(vectors[ctrl_lbl][i] == tvec[i] for i in sample) / len(sample)
        return pa - ca
    rng = random.Random(0)
    boots = sorted(gap([rng.choice(paired) for _ in paired]) for _ in range(2000)) if paired else [0]
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    point = gap(paired) if paired else float("nan")

    print(f"\n--- verdict ---")
    print(f"positive (R1-distill)     agrees with R1: {pos[2]:.3f}")
    print(f"control  (same-base Meta)  agrees with R1: {ctrl[2]:.3f}   <- ancestry confound")
    print(f"independent floor          agrees with R1: {ind_mean:.3f}   <- convergent alignment")
    print(f"\npaired positive-minus-control gap: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(n_paired={len(paired)})")
    beats_ancestry = lo > 0                      # CI excludes zero
    beats_floor = pos[2] > ind_mean
    print(f"SIGNAL?  distill aligns to R1 above the SAME-BASE control (CI>0) : "
          f"{'YES' if beats_ancestry else 'no'}")
    print(f"         distill aligns to R1 above the independent floor        : "
          f"{'YES' if beats_floor else 'no'}  (+{pos[2]-ind_mean:+.3f})")
    if beats_ancestry and beats_floor:
        print("\n=> PASS (pilot): the R1 distillation leaves a black-box behavioural fingerprint")
        print("   that clears the same-base ancestry confound. Scale the battery + panel.")
    else:
        print("\n=> No clean separation at pilot scale. Inspect per-item splits before judging;")
        print("   may need a larger/ sharper contested battery or K>1 sampling.")

    out = os.path.join(os.path.dirname(__file__), "..",
                       "scratch", "passive_or_raw.json")
    json.dump({"vectors": vectors, "contested": contested,
               "raw": {f"{k[0]}|{k[1]}": v for k, v in raw.items()}},
              open(out, "w"), indent=1)
    print(f"\ncalls={cli.calls}  spent=${cli.spent:.4f}  raw responses -> {out}")


if __name__ == "__main__":
    main()
