"""Leak-free observations for a computation-allocation controller."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


CANDIDATE_FEATURES = (
    "score",
    "score_gap",
    "depth",
    "remaining_depth",
    "cumulative_nodes",
    "last_nodes",
    "cumulative_time",
    "last_score_delta",
    "exact_bound",
    "lower_bound",
    "upper_bound",
    "currently_selected",
    "from_square",
    "to_square",
    "promotion",
    "has_been_searched",
    "budget_remaining",
    "action_legal",
)

GLOBAL_FEATURES = (
    "budget_remaining",
    "initial_budget",
    "branch_count",
    "mean_score",
    "best_score",
    "worst_score",
    "top_score_gap",
    "mean_depth",
    "max_depth",
    "total_nodes",
    "total_time",
    "decision_changes",
)


@dataclass(frozen=True)
class BranchView:
    move: str
    score: int
    depth: int
    cumulative_nodes: int = 0
    last_nodes: int = 0
    cumulative_time_ms: int = 0
    last_score_delta: int = 0
    bound: str | None = None
    searched: bool = False
    legal: bool = True


@dataclass(frozen=True)
class ControllerObservation:
    """Controller input without reference values or teacher actions."""

    moves: tuple[str, ...]
    candidates: tuple[tuple[float, ...], ...]
    global_features: tuple[float, ...]
    action_mask: tuple[bool, ...]

    @property
    def stop_index(self) -> int:
        return len(self.moves)


def _score(value: int) -> float:
    return math.tanh(value / 1_000.0)


def _count(value: int, scale: float) -> float:
    return min(math.log1p(max(value, 0)) / scale, 1.0)


def _square(name: str) -> float:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        return 0.0
    return (ord(name[0]) - ord("a") + 8 * (ord(name[1]) - ord("1"))) / 63.0


def make_observation(
    branches: Sequence[BranchView],
    remaining_budget: int,
    initial_budget: int,
    max_branch_depth: int,
    total_nodes: int,
    total_time_ms: int,
    decision_changes: int,
) -> ControllerObservation:
    if not branches:
        raise ValueError("a controller observation needs at least one root branch")
    if initial_budget < 0 or remaining_budget < 0 or remaining_budget > initial_budget:
        raise ValueError("invalid computation budget")

    ordered = tuple(sorted(branches, key=lambda branch: branch.move))
    best_score = max(branch.score for branch in ordered)
    sorted_scores = sorted((branch.score for branch in ordered), reverse=True)
    top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
    selected_move = max(ordered, key=lambda branch: (branch.score, branch.move)).move
    budget_ratio = remaining_budget / max(initial_budget, 1)
    max_depth = max(max_branch_depth, 1)

    candidates = []
    legal_actions = []
    for branch in ordered:
        promotion = " nbrq".find(branch.move[4]) / 4.0 if len(branch.move) == 5 else 0.0
        bound = branch.bound
        candidates.append(
            (
                _score(branch.score),
                _score(branch.score - best_score),
                branch.depth / max_depth,
                max(max_branch_depth - branch.depth, 0) / max_depth,
                _count(branch.cumulative_nodes, 20.0),
                _count(branch.last_nodes, 16.0),
                _count(branch.cumulative_time_ms, 12.0),
                _score(branch.last_score_delta),
                float(bound is None or bound == "exact"),
                float(bound == "lowerbound"),
                float(bound == "upperbound"),
                float(branch.move == selected_move),
                _square(branch.move[:2]),
                _square(branch.move[2:4]),
                promotion,
                float(branch.searched),
                budget_ratio,
                float(branch.legal and remaining_budget > 0),
            )
        )
        legal_actions.append(branch.legal and remaining_budget > 0)

    scores = [branch.score for branch in ordered]
    depths = [branch.depth for branch in ordered]
    global_features = (
        budget_ratio,
        min(initial_budget / 64.0, 1.0),
        min(len(ordered) / 64.0, 1.0),
        _score(round(sum(scores) / len(scores))),
        _score(best_score),
        _score(min(scores)),
        _score(top_gap),
        sum(depths) / len(depths) / max_depth,
        max(depths) / max_depth,
        _count(total_nodes, 22.0),
        _count(total_time_ms, 14.0),
        min(decision_changes / max(initial_budget, 1), 1.0),
    )

    # STOP is always a legal terminating action, including at zero budget.
    return ControllerObservation(
        moves=tuple(branch.move for branch in ordered),
        candidates=tuple(candidates),
        global_features=global_features,
        action_mask=tuple(legal_actions) + (True,),
    )
