"""Torch/transformers plumbing the interfaces call — device, tokenizer, the
mask-aware training loop, and trap-prefix scoring. Refactored verbatim (behavior-
preserving) from the validated spike rigs (`scratch/`); the science is unchanged,
only the packaging. Stringification stays at the CLI boundary — this module returns
values, never prints."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def pad_batch(tok, texts: list[str], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed [n, seq_len] ids + attention mask (eos-padded)."""
    ids = torch.full((len(texts), seq_len), tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(texts), seq_len), dtype=torch.long)
    for i, t in enumerate(texts):
        toks = tok(t)["input_ids"][:seq_len]
        ids[i, : len(toks)] = torch.tensor(toks)
        mask[i, : len(toks)] = 1
    return ids, mask


def train(base_name: str, ids: torch.Tensor, mask: torch.Tensor,
          epochs: int, batch: int, lr: float, seed: int):
    """Mask-aware fine-tune; padded positions excluded from loss."""
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(base_name).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(ids))
        for i in range(0, len(ids), batch):
            idx = perm[i : i + batch]
            b, m = ids[idx].to(DEVICE), mask[idx].to(DEVICE)
            labels = b.masked_fill(m == 0, -100)
            loss = model(input_ids=b, attention_mask=m, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def load_model(path: str):
    return AutoModelForCausalLM.from_pretrained(path).to(DEVICE).eval()


def trap_stats(model, tok, traps: list[tuple[str, str, str]], batch: int = 64) -> tuple[int, float]:
    """(greedy verbatim hits, mean teacher-forced logprob of the keyed target)
    over a trap set. traps are (sentence, prefix, target) tuples."""
    hits = 0
    logps: list[float] = []
    with torch.no_grad():
        for i in range(0, len(traps), batch):
            chunk = traps[i : i + batch]
            enc = tok([p for _, p, _ in chunk], return_tensors="pt",
                      padding=True, padding_side="left").to(DEVICE)
            out = model.generate(**enc, do_sample=False, max_new_tokens=8,
                                 pad_token_id=tok.eos_token_id)
            comps = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            hits += sum(t[2].strip() in c for t, c in zip(chunk, comps))
            for _, prefix, target in chunk:
                pre = tok(prefix, return_tensors="pt")["input_ids"].to(DEVICE)
                tgt = tok(target, return_tensors="pt")["input_ids"].to(DEVICE)
                full = torch.cat([pre, tgt], dim=1)
                logits = model(full).logits[0, pre.shape[1] - 1 : -1]
                lp = torch.log_softmax(logits.float(), dim=-1)
                logps.append(lp[torch.arange(tgt.shape[1]), tgt[0]].sum().item())
    return hits, float(np.mean(logps))
