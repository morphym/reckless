from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from controller_model import ControllerConfig, MaskedActorCritic, batch_observations
    from controller_state import BranchView, make_observation
    from online_train import anneal, boltzmann_choice


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class ControllerModelTests(unittest.TestCase):
    def observation(self, count, budget=2):
        branches = [
            BranchView(move=f"a{index + 1}a{index + 2}", score=index * 10, depth=1, legal=index != count - 1)
            for index in range(count)
        ]
        return make_observation(branches, budget, budget, 4, 0, 0, 0)

    def test_variable_frontier_and_masked_logits(self):
        observations = [self.observation(2), self.observation(3)]
        batch = batch_observations(observations, "cpu")
        model = MaskedActorCritic(ControllerConfig())
        logits, values = model(batch)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertLess(float(logits[0, 2].detach()), -1.0e30)
        self.assertTrue(torch.isfinite(logits[:, -1]).all())

    def test_parameter_size_is_small_but_nontrivial(self):
        model = MaskedActorCritic(ControllerConfig())
        count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(count, 100_000)
        self.assertLess(count, 1_000_000)

    def test_temperature_schedule_and_move_sampling(self):
        self.assertAlmostEqual(anneal(4.0, 1.0, 0.5), 2.0)
        rng = __import__("random").Random(91)
        choices = [boltzmann_choice(("a", "b"), [0, 100], 200.0, rng) for _ in range(100)]
        self.assertGreater(choices.count("b"), choices.count("a"))


if __name__ == "__main__":
    unittest.main()
