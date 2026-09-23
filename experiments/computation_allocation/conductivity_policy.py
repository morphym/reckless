"""External conductivity head and differentiable conserved tree flow."""
import math

import chess
import torch
from torch import nn


FEATURE_VERSION = 2  # Quiescence-scaled observed values replace raw static NNUE.
# Board (12 x 64), side to move, castling, ep file, move endpoints/promotion,
# then observed branch statistics. Never includes deeper reference labels.
INPUT_DIM = 768 + 1 + 4 + 8 + 64 + 64 + 5 + 10


def features(board, move, stats):
    x = [0.] * INPUT_DIM
    for square, piece in board.piece_map().items():
        plane = (0 if piece.color else 6) + piece.piece_type - 1
        x[plane * 64 + square] = 1.
    x[768] = 1. if board.turn else -1.
    x[769:773] = [float(board.has_kingside_castling_rights(chess.WHITE)),
                  float(board.has_queenside_castling_rights(chess.WHITE)),
                  float(board.has_kingside_castling_rights(chess.BLACK)),
                  float(board.has_queenside_castling_rights(chess.BLACK))]
    if board.ep_square is not None:
        x[773 + chess.square_file(board.ep_square)] = 1.
    x[781 + move.from_square] = 1.
    x[845 + move.to_square] = 1.
    x[909 + (0 if move.promotion is None else move.promotion - 1)] = 1.
    if len(stats) != 10:
        raise ValueError('expected ten observed branch statistics')
    x[914:] = stats
    return x


class ConductivityHead(nn.Module):
    """Shared edge encoder with permutation-equivariant sibling context."""
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        h = self.encoder(x)
        context = h.mean(0, keepdim=True).expand_as(h)
        logits = self.output(torch.cat((h, context), -1)).squeeze(-1)
        # Strictly positive, bounded sibling conductivities.
        return .01 + logits.softmax(0)


def conserved_flow(children, conductivity, frontier):
    """Unit source current, equal-pressure frontier sinks, unit edge lengths.

    Closed leaves have zero conductance. All operations retain autograd.
    Node indices must be topological (parent precedes child).
    """
    effective, branch = {}, {}
    for node in reversed(range(len(children))):
        if node in frontier:
            effective[node] = None  # Infinite sink conductance.
        else:
            terms = []
            for child in children[node]:
                g, d = effective[child], conductivity[child]
                branch[child] = d if g is None else d * g / (d + g)
                terms.append(branch[child])
            effective[node] = sum(terms)
    current = {0: next(iter(conductivity.values())).new_tensor(1.)}
    for node in range(len(children)):
        if node in frontier or not children[node]:
            continue
        total = effective[node]
        if not torch.is_tensor(total) or total.detach().item() <= 0:
            continue
        for child in children[node]:
            current[child] = current.get(node, total * 0) * branch[child] / total
    return {node: current[node] for node in sorted(frontier)}


def score_utility(info, scale=600.):
    """Bounded utility; mate sentinels must never masquerade as centipawns."""
    if info.score_kind == 'mate':
        return 1. if info.score_raw > 0 else -1.
    return math.tanh(info.score_raw / scale)
