import sys
from pathlib import Path
import unittest
import tempfile
import random
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
import torch
from conductivity_policy import ConductivityHead, INPUT_DIM, conserved_flow, features, score_utility
from train_conductivity import regret_loss, search, terminal_value, save_checkpoint, reference_values, write_tensorboard_update
from conductivity_positions import board_state, restore_board, FenSource, held_out
from reckless_uci import QuiescenceInfo


class FakeEngine:
    def new_game(self):
        pass

    def legal_moves(self, fen):
        return tuple(m.uci() for m in chess.Board(fen).legal_moves)

    def quiescence_evaluate(self, positions, max_nodes=4096):
        scores = []
        for fen, moves in positions:
            b = restore_board({'fen': fen, 'moves': moves})
            cp = 10 * sum((1 if p.color == b.turn else -1) * p.piece_type for p in b.piece_map().values())
            scores.append(QuiescenceInfo('cp', cp, 1))
        return scores


class ConductivityTests(unittest.TestCase):
    def test_features_and_permutation(self):
        b = chess.Board()
        x = torch.tensor([features(b, m, [0.] * 10) for m in b.legal_moves])
        self.assertEqual(x.shape, (20, INPUT_DIM))
        net = ConductivityHead(8)
        d = net(x)
        self.assertTrue(torch.all(d > 0))
        torch.testing.assert_close(net(x.flip(0)), d.flip(0))

    def test_flow_conservation_and_gradient(self):
        ds = {i: torch.tensor(float(i), requires_grad=True) for i in range(1, 5)}
        flow = conserved_flow([[1, 2], [3, 4], [], [], []], ds, {2, 3, 4})
        torch.testing.assert_close(sum(flow.values()), torch.tensor(1.))
        flow[3].log().backward()
        self.assertTrue(all(torch.isfinite(d.grad) for d in ds.values()))
        self.assertGreater(ds[3].grad.item(), 0)

    def test_closed_branch_has_no_current(self):
        ds = {i: torch.tensor(1.) for i in (1, 2)}
        flow = conserved_flow([[1, 2], [], []], ds, {1})
        self.assertEqual(flow[1].item(), 1.)

    def test_regret_gradient_favors_lower_cost(self):
        logits = torch.tensor([0., 0.], requires_grad=True)
        logs = list(logits.log_softmax(0))
        regret_loss([0., 1.], logs).backward()
        self.assertLess(logits.grad[0].item(), 0.)
        self.assertGreater(logits.grad[1].item(), 0.)

    def test_mate_and_terminal(self):
        self.assertEqual(score_utility(SimpleNamespace(score_kind='mate', score_raw=3)), 1.)
        self.assertEqual(score_utility(SimpleNamespace(score_kind='mate', score_raw=-3)), -1.)
        b = chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 0 1')
        self.assertEqual(terminal_value(b, chess.BLACK), -1.)

    def test_bound_reference_researched_and_pov_reversed(self):
        class Reference:
            def legal_moves(self, fen):
                return ('e2e4',)

            def analyze(self, fen, depth, multipv=1, moves=()):
                return SimpleNamespace(infos=[SimpleNamespace(
                    pv=('e2e4',), bound='lowerbound' if depth == 5 else None,
                    depth=depth, score_kind='cp', score_raw=-100)])
        values = reference_values(Reference(), chess.Board(), 5)
        self.assertGreater(values['e2e4'], 0)

    def test_equal_regret_exactly_zero_gradient(self):
        logits = torch.zeros(3, requires_grad=True)
        regret_loss([.1549840118336011] * 3, list(logits.log_softmax(0))).backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_search_budget_and_backward(self):
        torch.manual_seed(7)
        head = ConductivityHead(8)
        move, logp, stats = search(head, FakeEngine(), chess.Board(), 64, 3)
        self.assertIn(chess.Move.from_uci(move), chess.Board().legal_moves)
        self.assertLessEqual(stats['frontier_evaluations'], 64)
        self.assertGreater(stats['qsearch_nodes'], 0)
        self.assertGreater(stats['flow_steps'], 0)
        logp.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_progressive_evidence_changes_conductivity(self):
        torch.manual_seed(13)
        head = ConductivityHead(8)
        b = chess.Board()
        x = torch.tensor([features(b, m, [0.] * 10) for m in b.legal_moves])
        before = head(x)
        x[0, -7] = .9  # Current backed-up child value, not teacher value.
        self.assertFalse(torch.equal(before, head(x)))

    def test_checkpoint_roundtrip(self):
        net = ConductivityHead(8)
        optimizer = torch.optim.Adam(net.parameters())
        rng = random.Random(42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_checkpoint(path, net, optimizer, 3, SimpleNamespace(seed=42), rng)
            state = torch.load(path, weights_only=False)
            other = ConductivityHead(state['width'])
            other.load_state_dict(state['model'])
            self.assertEqual(state['update'], 3)
            for a, b in zip(net.parameters(), other.parameters()):
                torch.testing.assert_close(a, b)

    def test_history_roundtrip_preserves_repetition(self):
        board = chess.Board()
        for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 4:
            board.push_uci(move)
        restored = restore_board(board_state(board))
        self.assertEqual(restored.fen(), board.fen())
        self.assertEqual(restored.outcome().termination, chess.Termination.FIVEFOLD_REPETITION)

    def test_fen_source_ignores_wdl_and_skips_terminal(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fens.json'
            path.write_text(json.dumps([
                {'fen': '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', 'wdl': 'DO NOT READ'},
                {'fen': chess.STARTING_FEN, 'wdl': 'DO NOT READ'},
                {'fen': chess.STARTING_FEN, 'wdl': None},
            ]))
            args = SimpleNamespace(positions=path, evaluate_only=False)
            source = FenSource(args)
            self.assertEqual(source.next_board().fen(), chess.STARTING_FEN)
            resumed = FenSource(args, consumed=source.consumed)
            self.assertEqual(resumed.next_board().fen(), chess.STARTING_FEN)
            self.assertEqual(resumed.consumed, 3)

    def test_holdout_ignores_clocks(self):
        self.assertEqual(held_out(chess.STARTING_FEN), held_out(chess.STARTING_FEN.rsplit(' ', 2)[0] + ' 9 80'))

    def test_tensorboard_metrics_and_terminal_game(self):
        class Writer:
            def __init__(self):
                self.scalars = {}
                self.flushed = False

            def add_scalar(self, tag, value, step):
                self.scalars[tag] = (value, step)

            def flush(self):
                self.flushed = True

        writer = Writer()
        row = dict(mean_regret=.1, regret_std=.05, optimal_fraction=.5,
                   gradient_norm=.02, seconds=4., reference_seconds=1.,
                   search_seconds=2., optimizer_seconds=.1,
                   frontier_evaluations=32., qsearch_nodes=64.,
                   qsearch_truncated=2., qsearch_truncation_rate=.0625,
                   flow_steps=9., tree_nodes=40., game_ply=14,
                   game_result='1/2-1/2')
        write_tensorboard_update(writer, row, 7, False)
        self.assertEqual(writer.scalars['train/mean_regret'], (.1, 7))
        self.assertEqual(writer.scalars['train/search/qsearch_truncation_rate'], (.0625, 7))
        self.assertEqual(writer.scalars['train/game/result_white'], (0, 7))
        self.assertTrue(writer.flushed)


if __name__ == '__main__':
    unittest.main()
