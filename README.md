# assay

**A neutral testbed for model distillation attribution.** Measure whether a suspect model
was distilled from a given teacher's keyed outputs — with a calibrated false-positive rate,
under adversarial laundering. It measures radioactivity; it accuses nobody.

Marking schemes (watermarks, trap-streets) are *inputs*. assay owns the measurement layer:
plug in a **marker**, an **attack**, and a **detector**, and get comparable, calibrated numbers.

## Why

When a model is distilled from another's outputs, the mark left behind is *radioactive* — a
model trained on marked text inherits a detectable, keyed bias. The open question isn't
detection but **attribution**: proving *which* teacher a suspect was trained on, at a bounded
false-positive rate, even after the adversary paraphrases ("launders") the harvested text.
assay is the reproducible bench that measures exactly that, across (marker × attack) conditions.

## The result so far

A small-scale spike (82M–355M models) validated the core bet and stress-tested it:

- **Trap-street markers survive laundering.** Keyed fabricated facts (e.g. a bridge with a
  planted completion year) transfer through paraphrase-then-distill (**10.4σ**) where
  token-level watermarks die (**4.4σ**) — because the mark is the *claim*, not the words.
- **Attribution works.** A 4-teacher paternity test named the true teacher every time, with
  no false accusation (true keys ≥ 9.8σ, worst wrong key 1.0σ).
- **Calibration is honest.** An uncalibrated detector convicts an innocent model at raw
  z ≈ 25; a decoy-key empirical null fixes it — the innocent reads ~0.7σ.
- **Adversarial gauntlet:** 5 clean passes + 1 documented robustness limit (heavy continued
  fine-tuning). Full record in [`research-log.md`](research-log.md).

## Status

Graduated through a 5-gate qualification (kill-screen → spike → spec → scope → verify).
Building the **walking skeleton** now: `RunSpec → mark → distill → detect → calibrated verdict`.
The contract is in [`SPEC.md`](SPEC.md); the demo hook is a *model paternity test* CLI.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets numpy
# (skeleton CLI lands here — see SPEC.md § User Interface)
```

## Layout

| Path | What |
|---|---|
| [`SPEC.md`](SPEC.md) | The contract — interfaces, acceptance criteria, decisions |
| [`ROADMAP.md`](ROADMAP.md) | Where it's heading — passive, no-injection attribution (the bet) |
| [`research-log.md`](research-log.md) | The full experimental trail and every measured number |
| `scratch/` | The throwaway spike rigs the skeleton refactors from |

Not a watermark, not a prevention tool, not an accusation engine — a measurement bench.
Small-scale by design; frontier extrapolation is flagged, never claimed.
