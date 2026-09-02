"""CAMS-v0 -- Compositional Allocation and Merge Stream.

A synthetic continual stream whose *module structure is known by construction*. This is
the property the literature audit found missing: without a ground-truth skill count
``K*``, over- and under-allocation can only be inferred from downstream accuracy, and a
merge can never be scored as right or wrong.

Construction
------------
* A frozen random projection ``P`` plays the role of a frozen backbone; features are
  ``phi = tanh(x P^T)``. No arm may modify it.
* ``K*`` latent skills, each an input region ``mu_k`` plus a linear teacher ``W_k`` over
  the features. Labels are ``argmax(phi W_k)``.
* Segments are drawn from three kinds:
  ``novel``     -- a skill appearing for the first time (a spawn here is correct);
  ``recur``     -- an exact return of a previously seen skill (reuse should occur, no spawn);
  ``near_dup``  -- a perturbed variant of a seen skill (spawning then merging is correct,
                   and merging it with an unrelated skill is wrong).

``K*`` counts distinct skills; near-duplicates do not increase it. A policy that ends with
more than ``K*`` live modules has over-allocated, whatever its accuracy.

The stream is generated from a seed and never re-tuned after seeing which arm wins. See
docs/BENCHMARK-POLICY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

SegmentKind = Literal["novel", "recur", "near_dup"]


@dataclass
class StreamConfig:
    k_star: int = 6
    n_segments: int = 18
    d_in: int = 24
    d_feat: int = 48
    n_class: int = 4
    n_train_per_segment: int = 320
    n_eval_per_segment: int = 200
    chunk: int = 80
    recur_prob: float = 0.30
    near_dup_prob: float = 0.30
    perturb: float = 0.20
    region_scale: float = 2.2
    noise: float = 0.55
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Segment:
    index: int
    skill: int          # ground-truth skill id; near-duplicates share their parent's id
    kind: SegmentKind
    is_novel_onset: bool
    X: np.ndarray = field(repr=False)
    y: np.ndarray = field(repr=False)
    Xe: np.ndarray = field(repr=False)
    ye: np.ndarray = field(repr=False)


@dataclass
class Stream:
    cfg: StreamConfig
    P: np.ndarray = field(repr=False)
    segments: list[Segment] = field(repr=False)
    k_star: int
    novel_onsets: list[int]

    def features(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.P.T)

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def make_stream(cfg: StreamConfig) -> Stream:
    rng = np.random.default_rng(cfg.seed)
    P = rng.normal(scale=1.0 / np.sqrt(cfg.d_in), size=(cfg.d_feat, cfg.d_in))

    mus = rng.normal(scale=cfg.region_scale, size=(cfg.k_star, cfg.d_in))
    Ws = rng.normal(size=(cfg.k_star, cfg.d_feat, cfg.n_class))

    order: list[tuple[int, SegmentKind]] = []
    introduced: list[int] = []
    next_new = 0
    for _ in range(cfg.n_segments):
        u = rng.random()
        can_recur = len(introduced) > 0
        if next_new < cfg.k_star and (not can_recur or u > cfg.recur_prob + cfg.near_dup_prob):
            order.append((next_new, "novel"))
            introduced.append(next_new)
            next_new += 1
        elif can_recur and u < cfg.recur_prob:
            order.append((int(rng.choice(introduced)), "recur"))
        elif can_recur:
            order.append((int(rng.choice(introduced)), "near_dup"))
        else:  # pragma: no cover - only reachable if k_star exhausted with none introduced
            order.append((int(rng.choice(introduced)), "recur"))

    segments: list[Segment] = []
    novel_onsets: list[int] = []
    for i, (k, kind) in enumerate(order):
        mu = mus[k].copy()
        W = Ws[k].copy()
        if kind == "near_dup":
            mu = mu + rng.normal(scale=cfg.perturb * cfg.region_scale, size=cfg.d_in)
            W = W + rng.normal(scale=cfg.perturb, size=W.shape)

        def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
            X = mu + rng.normal(scale=cfg.noise, size=(n, cfg.d_in))
            phi = np.tanh(X @ P.T)
            y = np.argmax(phi @ W, axis=1)
            return X, y

        X, y = draw(cfg.n_train_per_segment)
        Xe, ye = draw(cfg.n_eval_per_segment)
        is_onset = kind == "novel"
        if is_onset:
            novel_onsets.append(i)
        segments.append(Segment(index=i, skill=k, kind=kind, is_novel_onset=is_onset,
                                X=X, y=y, Xe=Xe, ye=ye))

    return Stream(cfg=cfg, P=P, segments=segments, k_star=cfg.k_star,
                  novel_onsets=novel_onsets)


def onehot(y: np.ndarray, n_class: int) -> np.ndarray:
    out = np.zeros((y.size, n_class))
    out[np.arange(y.size), y] = 1.0
    return out


def stream_summary(stream: Stream) -> dict:
    kinds: dict[str, int] = {}
    for s in stream.segments:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    return {
        "n_segments": stream.n_segments,
        "k_star": stream.k_star,
        "kinds": kinds,
        "novel_onsets": list(stream.novel_onsets),
        "skill_sequence": [s.skill for s in stream.segments],
        "config": stream.cfg.as_dict(),
    }
