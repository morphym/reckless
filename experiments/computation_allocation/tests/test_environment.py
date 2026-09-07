from fractions import Fraction
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from allocation_env import BranchEstimate, RootBranchAllocationEnv, STOP
from cached_env import CachedCsEnv
from controller_state import CANDIDATE_FEATURES, GLOBAL_FEATURES
from diagnostic import (
    ALL_TOY_STATES,
    START,
    compatible_discounted_rewards,
    dense_rewards,
    discounted_return,
    make_solver,
    toy_model,
)
from live_env import LiveCsEnv
from reckless_uci import SearchInfo, SearchResult, normalize_score, parse_info


class ExactDiagnosticTests(unittest.TestCase):
    def test_kernel_and_budget_switch(self):
        model = toy_model()
        model.validate(ALL_TOY_STATES)
        solve = make_solver(model)
        self.assertEqual((solve(0, START).action, solve(0, START).expected_loss), (None, 10))
        self.assertEqual((solve(1, START).action, solve(1, START).expected_loss), ("investigate_B", 6))
        self.assertEqual((solve(2, START).action, solve(2, START).expected_loss), ("investigate_A", 2))
        self.assertEqual(solve(3, START).expected_loss, 2)

    def test_reliability_threshold(self):
        expected = {
            Fraction(0): ("investigate_B", 4),
            Fraction(1, 4): ("investigate_B", 4),
            Fraction(1, 2): ("investigate_A", 4),
            Fraction(3, 4): ("investigate_A", 2),
            Fraction(1): ("investigate_A", 0),
        }
        for probability, target in expected.items():
            decision = make_solver(toy_model(probability))(2, START)
            self.assertEqual((decision.action, decision.expected_loss), target)

    def test_reward_identities_and_discount_counterexample(self):
        losses = [Fraction(10), Fraction(10), Fraction(0)]
        rewards = dense_rewards(losses)
        self.assertEqual(sum(rewards), losses[0] - losses[-1])

        gamma = Fraction(1, 2)
        delayed = discounted_return(rewards, gamma)
        immediate = discounted_return(dense_rewards([Fraction(10), Fraction(4), Fraction(4)]), gamma)
        self.assertGreater(immediate, delayed)

        compatible = compatible_discounted_rewards(losses, gamma)
        self.assertEqual(
            discounted_return(compatible, gamma),
            losses[0] - gamma ** (len(losses) - 1) * losses[-1],
        )


class AllocationEnvironmentTests(unittest.TestCase):
    def make_env(self, budget=2):
        curves = {
            "a1a2": (
                BranchEstimate(1, 0, 1, 1),
                BranchEstimate(2, 0, 2, 2),
                BranchEstimate(3, 100, 3, 3),
            ),
            "b1b2": (
                BranchEstimate(1, 10, 1, 1),
                BranchEstimate(2, 20, 2, 2),
                BranchEstimate(3, 30, 3, 3),
            ),
        }
        return RootBranchAllocationEnv(curves, {"a1a2": 100, "b1b2": 30}, budget)

    def test_delayed_branch_is_budget_dependent(self):
        env = self.make_env()
        self.assertEqual(env.bellman_decision(0), (STOP, 70))
        self.assertEqual(env.bellman_decision(1), (STOP, 70))
        self.assertEqual(env.bellman_decision(2), ("a1a2", 0))

    def test_stop_is_terminal_and_free(self):
        env = self.make_env()
        result = env.step(STOP)
        self.assertTrue(result.terminated)
        self.assertEqual(result.remaining_budget, 2)
        with self.assertRaises(RuntimeError):
            env.step("a1a2")

    def test_rewards_telescope(self):
        env = self.make_env()
        initial = env.loss
        rewards = [env.step("a1a2").reward, env.step("a1a2").reward]
        self.assertEqual(sum(rewards), initial - env.loss)

    def test_controller_observation_masks_exhausted_branches(self):
        base = self.make_env(budget=2)
        env = CachedCsEnv(base.curves, base.reference_scores, budget=2)
        observation = env.observation()
        self.assertEqual(len(observation.candidates[0]), len(CANDIDATE_FEATURES))
        self.assertEqual(len(observation.global_features), len(GLOBAL_FEATURES))
        self.assertEqual(len(observation.action_mask), len(observation.moves) + 1)
        self.assertTrue(observation.action_mask[-1])


class MockReckless:
    def __init__(self):
        self.new_games = 0
        self.searches = []

    def new_game(self):
        self.new_games += 1

    def legal_moves(self, fen):
        return ("a2a3", "b2b3")

    def static_evaluate_after_moves(self, fen, moves):
        return [0, -10]

    def analyze_branch(self, fen, move, depth):
        self.searches.append((fen, move, depth))
        score = -120 if move == "a2a3" else -20
        info = SearchInfo(depth, 1, "cp", score, score, None, 123, 4, ("a7a6",))
        return SearchResult("a7a6", (info,))


class LiveEnvironmentTests(unittest.TestCase):
    def test_live_action_uses_native_session_and_telescopes(self):
        engine = MockReckless()
        env = LiveCsEnv(engine, "test fen", {"a2a3": 100, "b2b3": 30}, 2, (1, 2))
        self.assertEqual(engine.new_games, 1)
        self.assertEqual(env.selected_move, "b2b3")
        initial_loss = env.loss
        first = env.step("a2a3")
        second = env.step(STOP)
        self.assertEqual(engine.searches, [("test fen", "a2a3", 1)])
        self.assertEqual(first.reward + second.reward, initial_loss - env.loss)
        self.assertEqual(first.charged_nodes, 123)
        self.assertTrue(second.terminated)


class UciParsingTests(unittest.TestCase):
    def test_info_parser(self):
        info = parse_info("info depth 6 seldepth 9 multipv 2 score cp -17 nodes 1234 time 5 pv e2e4 e7e5")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual((info.depth, info.multipv, info.score, info.nodes, info.pv[0]), (6, 2, -17, 1234, "e2e4"))

    def test_mate_normalization(self):
        self.assertGreater(normalize_score("mate", 3), normalize_score("cp", 20_000))
        self.assertLess(normalize_score("mate", -3), normalize_score("cp", -20_000))

    def test_terminal_info_without_pv(self):
        info = parse_info("info depth 0 score mate 0")
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual((info.depth, info.score, info.pv), (0, -100_000, ()))


if __name__ == "__main__":
    unittest.main()
