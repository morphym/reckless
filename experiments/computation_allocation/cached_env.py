"""Controller adapter for cached Reckless branch curves."""

from __future__ import annotations

from typing import Mapping

from allocation_env import BranchEstimate, RootBranchAllocationEnv, STOP, StepResult
from controller_state import BranchView, ControllerObservation, make_observation


class CachedCsEnv:
    """Cheap CS environment used for PPO development before live rollouts."""

    def __init__(
        self,
        curves: Mapping[str, tuple[BranchEstimate, ...]],
        reference_scores: Mapping[str, int],
        budget: int,
    ) -> None:
        self.env = RootBranchAllocationEnv(curves, reference_scores, budget)
        self.decision_changes = 0
        self.max_branch_depth = max(estimate.depth for curve in curves.values() for estimate in curve)

    @classmethod
    def from_position(
        cls,
        position: Mapping[str, object],
        allocation_depths: tuple[int, ...],
        budget: int,
    ) -> "CachedCsEnv":
        raw_branches = position["branches"]
        assert isinstance(raw_branches, dict)
        curves = {
            move: tuple(
                BranchEstimate(
                    depth=depth,
                    score=int(per_depth[str(depth)]["score"]),
                    nodes=int(per_depth[str(depth)]["nodes"]),
                    time_ms=int(per_depth[str(depth)]["time_ms"]),
                    bound=per_depth[str(depth)].get("bound"),
                )
                for depth in allocation_depths
            )
            for move, per_depth in raw_branches.items()
        }
        reference_depth = str(position["reference_depth"])
        reference = {
            move: int(per_depth[reference_depth]["score"])
            for move, per_depth in raw_branches.items()
        }
        return cls(curves, reference, budget)

    @property
    def moves(self) -> tuple[str, ...]:
        return self.env.moves

    @property
    def selected_move(self) -> str:
        return self.env.selected_move

    @property
    def loss(self) -> int:
        return self.env.loss

    @property
    def terminated(self) -> bool:
        return self.env.terminated

    @property
    def legal_actions(self) -> tuple[str, ...]:
        return self.env.legal_actions

    def observation(self) -> ControllerObservation:
        views = []
        for index, move in enumerate(self.env.moves):
            curve = self.env.curves[move]
            revealed = self.env.indices[index]
            estimate = curve[revealed]
            previous_score = curve[revealed - 1].score if revealed else estimate.score
            views.append(
                BranchView(
                    move=move,
                    score=estimate.score,
                    depth=estimate.depth,
                    cumulative_nodes=sum(item.nodes for item in curve[1 : revealed + 1]),
                    last_nodes=estimate.nodes if revealed else 0,
                    cumulative_time_ms=sum(item.time_ms for item in curve[1 : revealed + 1]),
                    last_score_delta=estimate.score - previous_score,
                    bound=estimate.bound,
                    searched=revealed > 0,
                    legal=revealed + 1 < len(curve),
                )
            )
        return make_observation(
            views,
            self.env.remaining_budget,
            self.env.initial_budget,
            self.max_branch_depth,
            self.env.charged_nodes,
            self.env.charged_time_ms,
            self.decision_changes,
        )

    def action_for_index(self, index: int) -> str:
        observation = self.observation()
        if index == observation.stop_index:
            return STOP
        return observation.moves[index]

    def step(self, action: str) -> StepResult:
        previous = self.selected_move
        result = self.env.step(action)
        if self.selected_move != previous:
            self.decision_changes += 1
        return result
