#!/usr/bin/env python3
"""Distill deeper native Reckless PV flow into a pre-evidence conductivity head.

One native MultiPV search supplies root values and continuation lines. Root
flow is a softmax over those values; each continuation transports its root
flow to subsequent PV edges. If expanding a prefix would exceed the Physarum
frontier budget, that legal position starts a new supervised segment. No
Physarum rollout, REINFORCE baseline, or dataset WDL label is used.
"""
import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time

import chess
import torch

from conductivity_policy import ConductivityHead, FEATURE_VERSION, features, score_utility
from conductivity_positions import DEFAULT_REVISION, FenSource, board_position
from reckless_uci import RecklessUci


@dataclass
class Example:
    board: chess.Board
    moves: tuple[str, ...]
    target: tuple[float, ...]
    weight: float
    segment: int
    ply: int


def teacher_lines(engine, board, depth):
    """Return exact root values and PVs; never treat a bound as a label."""
    moves = engine.legal_moves(board.fen())
    if not moves:
        raise ValueError('teacher needs a nonterminal root')
    root_fen, history = board_position(board)
    result = engine.analyze(root_fen, depth, multipv=len(moves), moves=history)
    exact = {info.pv[0]: info for info in result.infos
             if info.pv and info.bound is None and info.depth >= depth}
    values, lines = {}, {}
    for move in moves:
        if move in exact:
            info = exact[move]
            values[move] = score_utility(info)
            lines[move] = tuple(info.pv)
            continue
        child = board.copy(stack=True)
        child.push_uci(move)
        outcome = child.outcome(claim_draw=False)
        if outcome is not None:
            values[move] = 0. if outcome.winner is None else (1. if outcome.winner == board.turn else -1.)
            lines[move] = (move,)
            continue
        child_root, child_history = board_position(child)
        retry = engine.analyze(child_root, depth - 1, moves=child_history)
        complete = [info for info in retry.infos if info.bound is None and info.depth >= depth - 1]
        if not complete:
            raise RuntimeError(f'no exact teacher value for {move}')
        values[move] = -score_utility(complete[0])
        lines[move] = (move, *complete[0].pv)
    return moves, values, lines


def softmax_values(moves, values, temperature):
    highest = max(values.values())
    exps = [math.exp((values[move] - highest) / temperature) for move in moves]
    total = sum(exps)
    return {move: value / total for move, value in zip(moves, exps)}


def flow_examples(board, moves, values, lines, budget, temperature, teacher_line_count):
    """Conserve teacher current down PVs and reset budget at split anchors."""
    if len(moves) > budget:
        raise ValueError('budget cannot expand all legal root moves')
    mass = softmax_values(moves, values, temperature)
    examples = [Example(board.copy(stack=True), tuple(moves), tuple(mass[m] for m in moves), 1., 0, 0)]
    splits = 0
    for root_move in sorted(moves, key=lambda move: values[move], reverse=True)[:teacher_line_count]:
        position = board.copy(stack=True)
        spent = len(moves)
        segment = 0
        for ply, move in enumerate(lines[root_move]):
            if ply == 0:
                position.push_uci(move)
                continue
            legal = tuple(m.uci() for m in position.legal_moves)
            if move not in legal or len(legal) > budget:
                break
            if spent + len(legal) > budget:
                spent = 0
                segment += 1
                splits += 1
            spent += len(legal)
            examples.append(Example(position.copy(stack=True), legal,
                                    tuple(float(m == move) for m in legal),
                                    mass[root_move], segment, ply))
            position.push_uci(move)
    return examples, splits, mass


def no_evidence_features(board, moves, max_depth):
    # Identical to the Rust expansion-time prior: no placeholder evaluation,
    # visit, or deposit is allowed to dilute the head's proposed conductivity.
    stats = [1. / max_depth, 0., 0., 0., 0., 0., 0., 1., 0., 0.]
    return [features(board, chess.Move.from_uci(move), stats) for move in moves]


def supervised_loss(head, examples, max_depth, device):
    total = None
    total_weight = 0.
    correct = 0.
    for example in examples:
        x = torch.tensor(no_evidence_features(example.board, example.moves, max_depth),
                         dtype=torch.float32, device=device)
        target = torch.tensor(example.target, dtype=torch.float32, device=device)
        # Conductivity has a fixed 0.01 exploration floor. Distill the
        # normalized learned part; the floor is restored by the model.
        predicted = (head(x) - .01).clamp_min(1e-12)
        loss = -(target * predicted.log()).sum()
        total = loss * example.weight if total is None else total + loss * example.weight
        total_weight += example.weight
        correct += example.weight * float(predicted.argmax() == target.argmax())
    return total / total_weight, correct / total_weight


def save_checkpoint(path, head, optimizer, update, args, rng, consumed):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
                'width': head.width, 'feature_version': FEATURE_VERSION,
                'objective': 'supervised-conductivity-v1', 'update': update,
                'config': vars(args), 'python_rng': rng.getstate(),
                'torch_rng': torch.get_rng_state(), 'consumed': consumed}, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True, help='native Reckless build')
    parser.add_argument('--output', type=Path, default=Path('outputs/conductivity_supervised'))
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--teacher-depth', type=int, default=12)
    parser.add_argument('--teacher-lines', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=.03)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--positions', type=Path, help='optional JSON list with fen fields')
    parser.add_argument('--dataset-split', default='strong+mid+low+early')
    parser.add_argument('--dataset-revision', default=DEFAULT_REVISION)
    parser.add_argument('--shuffle-buffer', type=int, default=1024)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.teacher_depth <= args.max_depth or args.max_depth < 2:
        parser.error('require teacher-depth > max-depth >= 2')
    if min(args.updates, args.budget, args.teacher_lines, args.width, args.shuffle_buffer) < 1:
        parser.error('all counts must be positive')
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('--temperature must be positive and finite')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    head = ConductivityHead(args.width).to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    start, consumed = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state.get('objective') != 'supervised-conductivity-v1':
            raise ValueError('resume checkpoint has a different training objective')
        if state['feature_version'] != FEATURE_VERSION or state['width'] != args.width:
            raise ValueError('checkpoint architecture mismatch')
        if not args.evaluate_only:
            for key in ('seed', 'dataset_split', 'dataset_revision', 'shuffle_buffer', 'positions',
                        'budget', 'max_depth', 'teacher_depth', 'teacher_lines', 'temperature'):
                if state['config'].get(key) != vars(args).get(key):
                    raise ValueError(f'resume changed {key}')
        head.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, consumed = state['update'], state['consumed']
        rng.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
    source = FenSource(args, consumed)
    args.output.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.output / 'tensorboard'))
    print(json.dumps({'event': 'training_started', 'objective': 'supervised-conductivity-v1',
                      'device': args.device, 'parameters': sum(p.numel() for p in head.parameters())}), flush=True)
    completed = start
    try:
        with RecklessUci(args.engine, args.timeout) as teacher:
            for update in range(start + 1, args.updates + 1):
                try:
                    board = source.next_board()
                except StopIteration:
                    print(json.dumps({'event': 'dataset_exhausted', 'consumed': source.consumed}), flush=True)
                    break
                begun = time.monotonic()
                reference_started = time.monotonic()
                moves, values, lines = teacher_lines(teacher, board, args.teacher_depth)
                reference_seconds = time.monotonic() - reference_started
                examples, splits, mass = flow_examples(board, moves, values, lines, args.budget,
                                                      args.temperature, args.teacher_lines)
                started = time.monotonic()
                with torch.set_grad_enabled(not args.evaluate_only):
                    loss, accuracy = supervised_loss(head, examples, args.max_depth, args.device)
                norm = torch.tensor(0.)
                if not args.evaluate_only:
                    optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('non-finite conductivity gradient')
                    optimizer.step()
                optimizer_seconds = time.monotonic() - started
                completed = update
                row = {'event': 'update_completed', 'update': update, 'loss': loss.item(),
                       'teacher_top1_accuracy': accuracy, 'teacher_entropy': -sum(p * math.log(p) for p in mass.values()),
                       'gradient_norm': norm.item(), 'examples': len(examples), 'budget_splits': splits,
                       'reference_seconds': reference_seconds, 'optimizer_seconds': optimizer_seconds,
                       'seconds': time.monotonic() - begun, 'fen': board.fen()}
                print(json.dumps(row), flush=True)
                with (args.output / 'metrics.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
                prefix = 'eval' if args.evaluate_only else 'train'
                for field in ('loss', 'teacher_top1_accuracy', 'teacher_entropy', 'gradient_norm',
                              'examples', 'budget_splits', 'reference_seconds', 'optimizer_seconds', 'seconds'):
                    writer.add_scalar(f'{prefix}/{field}', row[field], update)
                writer.flush()
                if not args.evaluate_only:
                    save_checkpoint(args.output / 'latest.pt', head, optimizer, update, args, rng, source.consumed)
    except KeyboardInterrupt:
        print('Interrupted; saving last completed update.', flush=True)
    finally:
        if not args.evaluate_only:
            save_checkpoint(args.output / 'latest.pt', head, optimizer, completed, args, rng, source.consumed)
        writer.close()
        source.close()


if __name__ == '__main__':
    main()
