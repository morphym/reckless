import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import torch

from conductivity_policy import ConductivityHead
from train_conductivity_supervised import flow_examples, no_evidence_features, supervised_loss


class SupervisedConductivityTests(unittest.TestCase):
    def test_teacher_flow_splits_without_fabricating_unseen_evidence(self):
        board = chess.Board()
        moves = tuple(move.uci() for move in board.legal_moves)
        values = {move: (1. if move == 'e2e4' else 0.) for move in moves}
        lines = {move: (move, 'e7e5') if move == 'e2e4' else (move,) for move in moves}
        examples, splits, mass = flow_examples(board, moves, values, lines, 25, .15, 1, 1)
        self.assertEqual(len(examples), 2)
        self.assertEqual(splits, 1)
        self.assertAlmostEqual(sum(mass.values()), 1.)
        self.assertEqual(examples[1].segment, 1)
        self.assertEqual(examples[1].board.fen().split()[1], 'b')
        self.assertEqual(examples[1].weight, mass['e2e4'])
        self.assertEqual(examples[1].target[examples[1].moves.index('e7e5')], 1.)
        x = no_evidence_features(board, ('e2e4',), 4)[0]
        self.assertEqual(x[914:], [.25, 0., 0., 0., 0., 0., 0., 1., 0., 0.])

    def test_supervised_loss_has_finite_gradient(self):
        board = chess.Board()
        moves = tuple(move.uci() for move in board.legal_moves)
        values = {move: (1. if move == 'e2e4' else 0.) for move in moves}
        lines = {move: (move,) for move in moves}
        examples, _, _ = flow_examples(board, moves, values, lines, 256, .15, 1, 4)
        head = ConductivityHead(8)
        loss, accuracy = supervised_loss(head, examples, 4, 'cpu')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
        self.assertGreaterEqual(accuracy, 0.)
        self.assertLessEqual(accuracy, 1.)


if __name__ == '__main__':
    unittest.main()
