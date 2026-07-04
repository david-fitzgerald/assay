"""CLI — the presentation boundary. All stringification lives here; the interface
modules return typed values. v0.1 skeleton: mark / attack / detect drive the spine
`RunSpec → mark → distill → detect → Verdict`. paternity / bench are v0.2."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .attack import run_attack
from .corpus import build_marked_corpus
from .detector import detect
from ._backend import tokenizer
from .runspec import RunSpec


def _cmd_mark(spec: RunSpec) -> int:
    tok = tokenizer(spec.teacher)
    ids, _ = build_marked_corpus(spec, tok)
    n_trap = spec.n_traps * spec.reps
    print(f"marked corpus: {len(ids)} seqs "
          f"({n_trap} trap injections, {n_trap / len(ids):.1%} fraction)")
    return 0


def _cmd_attack(spec: RunSpec) -> int:
    tok = tokenizer(spec.teacher)
    path = run_attack(spec, tok)
    print(f"student: {path}")
    return 0


def _cmd_detect(spec: RunSpec, suspect: str) -> int:
    tok = tokenizer(spec.teacher)
    if suspect in ("regen", "auto"):
        suspect = str(run_attack(spec, tok))  # build/reuse the student from the RunSpec
    v = detect(spec, suspect, tok)
    print(f"scheme={spec.scheme} key={spec.key} access={v.access_mode}")
    print(f"  calibrated σ = {v.calibrated_sigma:.2f}   (threshold {v.threshold})")
    print(f"  greedy hits  = {v.greedy_hits}/{spec.n_traps}")
    print(f"  decoy null   = {v.null_mu:.2f} ± {v.null_sd:.2f}  (D={v.decoy_D})")
    print(f"VERDICT: {'DETECTED' if v.detected else 'below threshold'}")
    return 0 if v.detected else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="assay", description="distillation-attribution testbed")
    p.add_argument("--version", action="version", version=f"assay {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("mark", "attack", "detect"):
        sp = sub.add_parser(name)
        sp.add_argument("--runspec", required=True)
        if name == "detect":
            sp.add_argument("--suspect", default="regen",
                            help="model path, or 'regen' to build from the RunSpec")
            sp.add_argument("--scheme")  # selectors; must match the RunSpec if given
            sp.add_argument("--key", type=int)
    args = p.parse_args(argv)

    spec = RunSpec.load(args.runspec)
    if getattr(args, "scheme", None) and args.scheme != spec.scheme:
        p.error(f"--scheme {args.scheme!r} != RunSpec scheme {spec.scheme!r}")
    if getattr(args, "key", None) is not None and args.key != spec.key:
        p.error(f"--key {args.key} != RunSpec key {spec.key}")

    if args.cmd == "mark":
        return _cmd_mark(spec)
    if args.cmd == "attack":
        return _cmd_attack(spec)
    return _cmd_detect(spec, args.suspect)


if __name__ == "__main__":
    sys.exit(main())
