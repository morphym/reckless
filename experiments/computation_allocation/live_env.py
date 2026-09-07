"""Live CS environment backed by persistent native Reckless searches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from allocation_env import STOP
from controller_state import BranchView, ControllerObservation, make_observation
from reckless_uci import RecklessUci


@dataclass
class LiveBranch:
    move: str
    score: int
    depth: int = 0
    searches: int = 0
    cumulative_nodes: int = 0
    last_nodes: int = 0
    cumulative_time_ms: int = 0
    last_score_delta: int = 0
    bound: str | None = None


@dataclass(frozen=True)
class LiveStep:
    action: str
    selected_move: str
    loss: int
    reward: int
    remaining_budget: int
    terminated: bool
    charged_nodes: int
    charged_time_ms: int
    reached_depth: int


class LiveCsEnv:
    """Allocate depth increments among root branches.

    Reckless owns legal move generation, child position construction, alpha-beta
    search, MovePicker ordering, NNUE evaluation, histories, and the TT.  This
    class owns only the higher-level computation action and reward boundary.
    Reference scores are used for reward/loss and never enter observations.
    """

    def __init__(
        self,
        engine: RecklessUci,
        fen: str,
        reference_scores: Mapping[str, int],
        budget: int,
        depth_schedule: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if not depth_schedule or any(depth <= 0 for depth in depth_schedule):
            raise ValueError("depth_schedule must contain positive depths")
        if tuple(sorted(set(depth_schedule))) != depth_schedule:
            raise ValueError("depth_schedule must be strictly increasing")

        self.engine = engine
        self.fen = fen
        self.depth_schedule = depth_schedule
        self.initial_budget = budget
        self.remaining_budget = budget
        self.terminated = False
        self.total_nodes = 0
        self.total_time_ms = 0
        self.decision_changes = 0

        engine.new_game()
        moves = engine.legal_moves(fen)
        if not moves:
            raise ValueError("terminal roots have no computation actions")
        if set(moves) != set(reference_scores):
            missing = sorted(set(moves) - set(reference_scores))
            extra = sorted(set(reference_scores) - set(moves))
            raise ValueError(f"reference mismatch: missing={missing}, extra={extra}")

        # The child is opponent-to-move, hence the sign inversion.
        static_scores = engine.static_evaluate_after_moves(fen, moves)
        self.branches = {
            move: LiveBranch(move=move, score=-child_score)
            for move, child_score in zip(moves, static_scores, strict=True)
        }
        self.reference_scores = dict(reference_scores)

    @classmethod
    def open(
        cls,
        executable: Path,
        fen: str,
        reference_scores: Mapping[str, int],
        budget: int,
        depth_schedule: tuple[int, ...] = (1, 2, 3, 4),
    ) -> "LiveCsEnv":
        return cls(RecklessUci(executable), fen, reference_scores, budget, depth_schedule)

    @property
    def selected_move(self) -> str:
        return max(self.branches, key=lambda move: (self.branches[move].score, move))

    @property
    def loss(self) -> int:
        best_reference = max(self.reference_scores.values())
        return best_reference - self.reference_scores[self.selected_move]

    @property
    def legal_actions(self) -> tuple[str, ...]:
        if self.terminated:
            return ()
        if self.remaining_budget == 0:
            return (STOP,)
        moves = tuple(
            move
            for move, branch in sorted(self.branches.items())
            if branch.searches < len(self.depth_schedule)
        )
        return moves + (STOP,)

    def action_for_index(self, index: int) -> str:
        observation = self.observation()
        if index == observation.stop_index:
            return STOP
        return observation.moves[index]

    def observation(self) -> ControllerObservation:
        views = [
            BranchView(
                move=branch.move,
                score=branch.score,
                depth=branch.depth,
                cumulative_nodes=branch.cumulative_nodes,
                last_nodes=branch.last_nodes,
                cumulative_time_ms=branch.cumulative_time_ms,
                last_score_delta=branch.last_score_delta,
                bound=branch.bound,
                searched=branch.searches > 0,
                legal=branch.searches < len(self.depth_schedule),
            )
            for branch in self.branches.values()
        ]
        return make_observation(
            views,
            self.remaining_budget,
            self.initial_budget,
            self.depth_schedule[-1],
            self.total_nodes,
            self.total_time_ms,
            self.decision_changes,
        )

    def step(self, action: str) -> LiveStep:
        if self.terminated:
            raise RuntimeError("episode has terminated")
        if action not in self.legal_actions:
            raise ValueError(f"illegal action: {action}")

        previous_loss = self.loss
        previous_move = self.selected_move
        charged_nodes = 0
        charged_time_ms = 0
        reached_depth = 0

        if action == STOP:
            self.terminated = True
        else:
            branch = self.branches[action]
            reached_depth = self.depth_schedule[branch.searches]
            result = self.engine.analyze_branch(self.fen, action, reached_depth)
            info = result.infos[0]
            old_score = branch.score
            branch.score = -info.score
            branch.depth = reached_depth
            branch.searches += 1
            branch.last_score_delta = branch.score - old_score
            branch.last_nodes = info.nodes
            branch.cumulative_nodes += info.nodes
            branch.cumulative_time_ms += info.time_ms
            branch.bound = {
                "lowerbound": "upperbound",
                "upperbound": "lowerbound",
            }.get(info.bound, info.bound)
            charged_nodes = info.nodes
            charged_time_ms = info.time_ms
            self.total_nodes += charged_nodes
            self.total_time_ms += charged_time_ms
            self.remaining_budget -= 1
            if self.selected_move != previous_move:
                self.decision_changes += 1
            if self.remaining_budget == 0:
                self.terminated = True

        current_loss = self.loss
        return LiveStep(
            action=action,
            selected_move=self.selected_move,
            loss=current_loss,
            reward=previous_loss - current_loss,
            remaining_budget=self.remaining_budget,
            terminated=self.terminated,
            charged_nodes=charged_nodes,
            charged_time_ms=charged_time_ms,
            reached_depth=reached_depth,
        )

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> "LiveCsEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
