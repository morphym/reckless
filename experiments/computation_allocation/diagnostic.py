"""Exact finite-horizon diagnostics from ``cs-complete.md``.

The module intentionally uses only exact rational arithmetic.  It is a small
executable specification for the parts of the paper that must be correct before
connecting a learned controller to a chess engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Generic, Hashable, Iterable, TypeVar


State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


@dataclass(frozen=True)
class Outcome(Generic[State]):
    probability: Fraction
    next_state: State


@dataclass(frozen=True)
class Decision(Generic[Action]):
    action: Action | None
    expected_loss: Fraction


@dataclass(frozen=True)
class FiniteModel(Generic[State, Action]):
    actions: Callable[[State], tuple[Action, ...]]
    transition: Callable[[State, Action], tuple[Outcome[State], ...]]
    stop_loss: Callable[[State], Fraction]

    def validate(self, states: Iterable[State]) -> None:
        for state in states:
            for action in self.actions(state):
                outcomes = self.transition(state, action)
                if not outcomes:
                    raise ValueError(f"empty transition: {state=}, {action=}")
                if any(outcome.probability < 0 for outcome in outcomes):
                    raise ValueError(f"negative probability: {state=}, {action=}")
                if sum((outcome.probability for outcome in outcomes), Fraction()) != 1:
                    raise ValueError(f"probabilities do not sum to one: {state=}, {action=}")


def make_solver(model: FiniteModel[State, Action]):
    """Return an at-most-budget Bellman solver with genuine optional stopping.

    STOP is represented by ``Decision.action is None`` and terminates now.  It
    is not a transition and cannot consume budget before later continuation.
    Strict comparison preserves deterministic action-list tie ordering.
    """

    @lru_cache(maxsize=None)
    def solve(budget: int, state: State) -> Decision[Action]:
        if budget < 0:
            raise ValueError("budget must be non-negative")

        best = Decision(action=None, expected_loss=model.stop_loss(state))
        if budget == 0:
            return best

        for action in model.actions(state):
            candidate = sum(
                (
                    outcome.probability
                    * solve(budget - 1, outcome.next_state).expected_loss
                    for outcome in model.transition(state, action)
                ),
                Fraction(),
            )
            if candidate < best.expected_loss:
                best = Decision(action=action, expected_loss=candidate)
        return best

    return solve


START = "start"
A_PRIMED = "a_primed"
B_ONE = "b_one"
B_TWO = "b_two"
SUCCESS = "success"
BAD_OUTCOME = "bad_outcome"
INVESTIGATE_A = "investigate_A"
INVESTIGATE_B = "investigate_B"


def toy_model(success_probability: Fraction = Fraction(3, 4)) -> FiniteModel[str, str]:
    def actions(state: str) -> tuple[str, ...]:
        return {
            START: (INVESTIGATE_A, INVESTIGATE_B),
            A_PRIMED: (INVESTIGATE_A,),
            B_ONE: (INVESTIGATE_B,),
        }.get(state, ())

    def deterministic(state: str) -> tuple[Outcome[str], ...]:
        return (Outcome(Fraction(1), state),)

    def transition(state: str, action: str) -> tuple[Outcome[str], ...]:
        if (state, action) == (START, INVESTIGATE_A):
            return deterministic(A_PRIMED)
        if (state, action) == (A_PRIMED, INVESTIGATE_A):
            return (
                Outcome(success_probability, SUCCESS),
                Outcome(1 - success_probability, BAD_OUTCOME),
            )
        if (state, action) == (START, INVESTIGATE_B):
            return deterministic(B_ONE)
        if (state, action) == (B_ONE, INVESTIGATE_B):
            return deterministic(B_TWO)
        raise ValueError(f"illegal action: {state=}, {action=}")

    losses = {
        START: 10,
        A_PRIMED: 10,
        B_ONE: 6,
        B_TWO: 4,
        SUCCESS: 0,
        BAD_OUTCOME: 8,
    }
    return FiniteModel(actions, transition, lambda state: Fraction(losses[state]))


ALL_TOY_STATES = (START, A_PRIMED, B_ONE, B_TWO, SUCCESS, BAD_OUTCOME)


def dense_rewards(losses: list[Fraction]) -> list[Fraction]:
    return [before - after for before, after in zip(losses, losses[1:])]


def discounted_return(rewards: list[Fraction], gamma: Fraction) -> Fraction:
    return sum((gamma**index * reward for index, reward in enumerate(rewards)), Fraction())


def compatible_discounted_rewards(losses: list[Fraction], gamma: Fraction) -> list[Fraction]:
    return [before - gamma * after for before, after in zip(losses, losses[1:])]


def run_diagnostics() -> dict[str, object]:
    model = toy_model()
    model.validate(ALL_TOY_STATES)
    solve = make_solver(model)

    budgets = {
        budget: {
            "action": solve(budget, START).action or "STOP",
            "expected_loss": str(solve(budget, START).expected_loss),
        }
        for budget in range(4)
    }

    sweep = {}
    for probability in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)):
        decision = make_solver(toy_model(probability))(2, START)
        sweep[str(probability)] = {
            "action": decision.action or "STOP",
            "expected_loss": str(decision.expected_loss),
        }

    losses = [Fraction(10), Fraction(10), Fraction(0)]
    rewards = dense_rewards(losses)

    gamma = Fraction(1, 2)
    delayed = discounted_return(dense_rewards([Fraction(10), Fraction(10), Fraction(0)]), gamma)
    immediate = discounted_return(dense_rewards([Fraction(10), Fraction(4), Fraction(4)]), gamma)
    compatible = discounted_return(compatible_discounted_rewards(losses, gamma), gamma)

    return {
        "kernel_valid": True,
        "budgets": budgets,
        "reliability_sweep": sweep,
        "dense_telescoping": {
            "rewards": [str(value) for value in rewards],
            "sum": str(sum(rewards, Fraction())),
            "initial_minus_terminal": str(losses[0] - losses[-1]),
        },
        "discount_counterexample": {
            "gamma": str(gamma),
            "delayed_terminal_zero_return": str(delayed),
            "immediate_terminal_four_return": str(immediate),
            "ordinary_discount_reverses_preference": immediate > delayed,
            "compatible_delayed_return": str(compatible),
            "compatible_identity_rhs": str(losses[0] - gamma ** (len(losses) - 1) * losses[-1]),
        },
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_diagnostics(), indent=2))
