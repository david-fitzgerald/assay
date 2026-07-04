"""Source + Marker: the teacher generates outputs (the SourceFixture), then the
scheme injects a keyed trace (the mark). Produces the marked corpus the student
distills on. Faithful to the threat model — the student trains on TEACHER outputs,
not arbitrary text."""

from __future__ import annotations

import torch

from ._backend import DEVICE, load_model, pad_batch, tokenizer
from .runspec import RunSpec
from .scheme import get_scheme

PROMPT_LEN, GEN_LEN = 8, 64
SEQ_LEN = PROMPT_LEN + GEN_LEN


def _wikitext_prefixes(tok, n: int, split: str) -> torch.Tensor:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    out = []
    for line in ds["text"]:
        line = line.strip()
        if len(line.split()) < 12 or line.startswith("="):
            continue
        ids = tok(line, return_tensors="pt")["input_ids"][0]
        if len(ids) >= PROMPT_LEN:
            out.append(ids[:PROMPT_LEN])
            if len(out) >= n:
                break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} usable prefixes in wikitext:{split}, wanted {n}")
    return torch.stack(out)


def _generate(teacher, prompts: torch.Tensor, seed: int, batch: int = 64) -> torch.Tensor:
    torch.manual_seed(seed)
    kw = dict(do_sample=True, top_k=0, max_new_tokens=GEN_LEN, min_new_tokens=GEN_LEN,
              pad_token_id=teacher.config.eos_token_id)
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = prompts[i : i + batch].to(DEVICE)
            out.append(teacher.generate(chunk, attention_mask=torch.ones_like(chunk), **kw).cpu())
    return torch.cat(out)


def build_marked_corpus(spec: RunSpec, tok) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-generated corpus + keyed trap sentences (× reps). Returns (ids, mask)."""
    teacher = load_model(spec.teacher)
    prompts = _wikitext_prefixes(tok, spec.n_corpus, spec.corpus_split)
    base = _generate(teacher, prompts, seed=spec.seed + 1)
    del teacher

    scheme = get_scheme(spec.scheme)
    trap_sents = scheme.sentences(spec.key, spec.n_traps) * spec.reps
    trap_ids, trap_mask = pad_batch(tok, trap_sents, SEQ_LEN)

    ids = torch.cat([base, trap_ids])
    mask = torch.cat([torch.ones_like(base), trap_mask])
    return ids, mask
