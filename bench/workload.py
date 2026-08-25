"""Workload generation.

Requests arrive as a Poisson process with a fixed seed, and each request draws a
prompt from the committed prompt set. Given the same spec and prompt file the
schedule is identical on every machine, which is the point: a run can be replayed
and diffed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASSES = ("short", "medium", "long")


@dataclass(frozen=True)
class PromptItem:
    id: str
    prompt_class: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class ScheduledRequest:
    seq: int
    arrival_offset_s: float
    prompt: PromptItem


@dataclass
class WorkloadSpec:
    seed: int
    arrival_rate_rps: float
    request_count: int
    class_weights: dict[str, float]

    def validate(self) -> None:
        if self.arrival_rate_rps <= 0:
            raise ValueError("arrival_rate_rps must be positive")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        unknown = set(self.class_weights) - set(CLASSES)
        if unknown:
            raise ValueError(f"unknown prompt classes: {sorted(unknown)}")
        total = sum(self.class_weights.get(c, 0.0) for c in CLASSES)
        if total <= 0:
            raise ValueError("class weights must sum to a positive value")


def load_prompts(path: str | Path) -> list[PromptItem]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["class"] not in CLASSES:
                raise ValueError(f"{path}:{line_no}: bad class {row['class']!r}")
            items.append(
                PromptItem(
                    id=row["id"],
                    prompt_class=row["class"],
                    prompt=row["prompt"],
                    max_tokens=int(row["max_tokens"]),
                )
            )
    if not items:
        raise ValueError(f"{path}: no prompts")
    return items


def poisson_arrivals(spec: WorkloadSpec) -> np.ndarray:
    """Cumulative arrival offsets in seconds for a seeded Poisson process."""
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    inter = rng.exponential(1.0 / spec.arrival_rate_rps, size=spec.request_count)
    return np.cumsum(inter)


def _class_probabilities(spec: WorkloadSpec) -> np.ndarray:
    weights = np.array([max(spec.class_weights.get(c, 0.0), 0.0) for c in CLASSES])
    return weights / weights.sum()


def build_requests(spec: WorkloadSpec, prompts: list[PromptItem]) -> list[ScheduledRequest]:
    """Pair each arrival offset with a prompt drawn by weighted class."""
    spec.validate()
    by_class: dict[str, list[PromptItem]] = {c: [] for c in CLASSES}
    for item in prompts:
        by_class[item.prompt_class].append(item)
    empty = [c for c in CLASSES if not by_class[c]]
    if empty:
        raise ValueError(f"prompt set has no items for classes: {empty}")

    rng = np.random.default_rng(spec.seed + 1)
    probs = _class_probabilities(spec)
    arrivals = poisson_arrivals(spec)
    out = []
    for seq, offset in enumerate(arrivals):
        cls = CLASSES[rng.choice(len(CLASSES), p=probs)]
        item = by_class[cls][int(rng.integers(len(by_class[cls])))]
        out.append(ScheduledRequest(seq=seq, arrival_offset_s=float(offset), prompt=item))
    return out
