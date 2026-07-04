"""RunSpec — the single versioned config artifact that owns all reproducibility
state (SPEC.md § Data Model). Every CLI command reads the section it needs; a run
is reproducible from its RunSpec alone. v0.1 skeleton: the sections the spine uses."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    teacher: str
    student_base: str
    corpus_split: str
    n_corpus: int
    scheme: str
    key: int
    n_traps: int
    reps: int
    epochs: int
    batch: int
    lr: float
    decoy_keys: list[int]
    threshold: float
    seed: int
    work_dir: Path = field(default=Path.home() / ".cache" / "assay")

    @staticmethod
    def load(path: str | Path) -> "RunSpec":
        raw = tomllib.loads(Path(path).read_text())
        m, mk, at, ev = raw["models"], raw["mark"], raw["attack"], raw["eval"]
        return RunSpec(
            teacher=m["teacher"],
            student_base=m["student_base"],
            corpus_split=raw["source"]["corpus_split"],
            n_corpus=raw["source"]["n_corpus"],
            scheme=mk["scheme"],
            key=mk["key"],
            n_traps=mk["n_traps"],
            reps=mk["reps"],
            epochs=at["epochs"],
            batch=at["batch"],
            lr=float(at["lr"]),
            decoy_keys=list(ev["decoy_keys"]),
            threshold=float(ev["threshold"]),
            seed=raw.get("seeds", {}).get("global", 0),
            work_dir=Path(raw.get("work_dir", Path.home() / ".cache" / "assay")).expanduser(),
        )

    @property
    def student_dir(self) -> Path:
        # Keyed by the config that determines the student, so a changed mark/attack
        # regenerates rather than silently reusing a stale student.
        tag = f"{self.scheme}_{self.key}_{self.n_traps}x{self.reps}_e{self.epochs}"
        return self.work_dir / f"student_{tag}"
