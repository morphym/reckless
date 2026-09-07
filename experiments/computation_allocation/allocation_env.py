"""A deterministic, pre-policy root-branch computation environment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping


STOP = "STOP"


@dataclass(frozen=True)
class BranchEstimate:
    depth: int
    score: int
    nodes: int
    time_ms: int
    bound: str | None = None


@dataclass(frozen=True)
class StepResult:
    action: str
    selected_move: str
    loss: int
    reward: int
    remaining_budget: int
    terminated: bool
    charged_nodes: int
    charged_time_ms: int


class RootBranchAllocationEnv:
    """Reveal progressively deeper Reckless estimates for chosen root moves.

    Each non-STOP action advances one root branch by one precomputed depth
    level.  The mixed-depth root decision is the greatest lexical move among
    tied maximum estimates, making selection total and deterministic.

    This environment is intentionally controller-free.  It validates state,
    action, budget, reward and STOP behavior before a policy head is added.
    """

    def __init__(
        self,
        curves: Mapping[str, tuple[BranchEstimate, ...]],
        reference_scores: Mapping[str, int],
        budget: int,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if not curves:
            raise ValueError("at least one legal root move is required")
        if set(curves) != set(reference_scores):
            raise ValueError("curves and reference scores must cover the same moves")
        if any(not curve for curve in curves.values()):
            raise ValueError("every move needs an initial estimate")

        self.moves = tuple(sorted(curves))
        self.curves = {move: tuple(curves[move]) for move in self.moves}
        self.reference_scores = dict(reference_scores)
        self.initial_budget = budget
        self.indices = [0] * len(self.moves)
        self.remaining_budget = budget
        self.terminated = False
        self.charged_nodes = 0
        self.charged_time_ms = 0

    def clone(self) -> "RootBranchAllocationEnv":
        clone = RootBranchAllocationEnv(self.curves, self.reference_scores, self.initial_budget)
        clone.indices = self.indices.copy()
        clone.remaining_budget = self.remaining_budget
        clone.terminated = self.terminated
        clone.charged_nodes = self.charged_nodes
        clone.charged_time_ms = self.charged_time_ms
        return clone

    def state_key(self) -> tuple[int, ...]:
        return tuple(self.indices)

    def estimate(self, move: str) -> BranchEstimate:
        return self.curves[move][self.indices[self.moves.index(move)]]

    def selected_move_for(self, indices: tuple[int, ...] | None = None) -> str:
        indices = self.state_key() if indices is None else indices
        return max(
            self.moves,
            key=lambda move: (self.curves[move][indices[self.moves.index(move)]].score, move),
        )

    def loss_for(self, indices: tuple[int, ...] | None = None) -> int:
        selected = self.selected_move_for(indices)
        return max(self.reference_scores.values()) - self.reference_scores[selected]

    @property
    def selected_move(self) -> str:
        return self.selected_move_for()

    @property
    def loss(self) -> int:
        return self.loss_for()

    def legal_actions_for(self, indices: tuple[int, ...] | None = None) -> tuple[str, ...]:
        indices = self.state_key() if indices is None else indices
        actions = [
            move
            for i, move in enumerate(self.moves)
            if indices[i] + 1 < len(self.curves[move])
        ]
        return tuple(actions) + (STOP,)

    @property
    def legal_actions(self) -> tuple[str, ...]:
        if self.terminated:
            return ()
        if self.remaining_budget == 0:
            return (STOP,)
        return self.legal_actions_for()

    def step(self, action: str) -> StepResult:
        if self.terminated:
            raise RuntimeError("episode has terminated")
        if action not in self.legal_actions:
            raise ValueError(f"illegal action: {action}")

        previous_loss = self.loss
        charged_nodes = 0
        charged_time_ms = 0

        if action == STOP:
            self.terminated = True
        else:
            if self.remaining_budget <= 0:
                raise ValueError("computation budget is exhausted; only STOP is legal")
            index = self.moves.index(action)
            self.indices[index] += 1
            estimate = self.curves[action][self.indices[index]]
            charged_nodes = estimate.nodes
            charged_time_ms = estimate.time_ms
            self.charged_nodes += charged_nodes
            self.charged_time_ms += charged_time_ms
            self.remaining_budget -= 1
            if self.remaining_budget == 0:
                self.terminated = True

        current_loss = self.loss
        return StepResult(
            action=action,
            selected_move=self.selected_move,
            loss=current_loss,
            reward=previous_loss - current_loss,
            remaining_budget=self.remaining_budget,
            terminated=self.terminated,
            charged_nodes=charged_nodes,
            charged_time_ms=charged_time_ms,
        )

    def bellman_decision(self, budget: int | None = None) -> tuple[str, int]:
        """Exact oracle for the deterministic cached environment.

        It is a diagnostic baseline, not an observation available to a future
        learned policy: terminal loss uses privileged reference scores.
        """

        remaining = self.remaining_budget if budget is None else budget
        start = self.state_key()

        @lru_cache(maxsize=None)
        def solve(indices: tuple[int, ...], units: int) -> tuple[str, int]:
            best_action = STOP
            best_loss = self.loss_for(indices)
            if units == 0:
                return best_action, best_loss

            for action in self.legal_actions_for(indices):
                if action == STOP:
                    continue
                move_index = self.moves.index(action)
                successor = list(indices)
                successor[move_index] += 1
                _, candidate_loss = solve(tuple(successor), units - 1)
                if candidate_loss < best_loss:
                    best_action, best_loss = action, candidate_loss
            return best_action, best_loss

        return solve(start, remaining)
