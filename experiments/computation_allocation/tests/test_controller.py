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
    from online_train import anneal, boltzmann_choice, write_tensorboard_update


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

    def test_tensorboard_update_logs_scalars_histograms_and_flushes(self):
        class RecordingWriter:
            def __init__(self):
                self.scalars = []
                self.histograms = []
                self.flushes = 0

            def add_scalar(self, tag, value, step):
                self.scalars.append((tag, value, step))

            def add_histogram(self, tag, value, step):
                self.histograms.append((tag, value, step))

            def flush(self):
                self.flushes += 1

        writer = RecordingWriter()
        row = {
            "update": 7,
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "entropy": 0.3,
            "mean_initial_loss": 12.0,
            "mean_terminal_loss": 7.0,
            "mean_return": 5.0,
            "mean_nodes": 100.0,
            "episodes": 2,
            "transitions": 6,
            "root_temperature_cp": 300.0,
            "controller_temperature": 2.0,
            "update_seconds": 4.0,
            "reference_seconds": 2.0,
            "reference_roots_per_second": 1.0,
            "rollout_seconds": 1.5,
            "ppo_seconds": 0.5,
            "episodes_per_second": 0.5,
            "all_rewards_telescope": True,
        }
        write_tensorboard_update(writer, row, [10, 14], [5, 9], [5.0, 5.0], [80, 120])

        self.assertEqual(len(writer.scalars), 18)
        self.assertEqual(len(writer.histograms), 4)
        self.assertTrue(all(item[2] == 7 for item in writer.scalars + writer.histograms))
        self.assertEqual(writer.flushes, 1)


if __name__ == "__main__":
    unittest.main()
