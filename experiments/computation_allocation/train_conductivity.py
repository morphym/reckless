#!/usr/bin/env python3
"""Train local conductivities through stochastic Physarum search, not imitation.

Prototype Python search, independent of the production Rust/UCI search.
Native Reckless supplies legal moves and quiescence evaluations. A SEPARATE
native engine supplies deeper root-move values used only after search finishes.
"""
import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import statistics
import time

import chess
import torch
from torch.distributions import Categorical

from conductivity_policy import ConductivityHead, FEATURE_VERSION, features, conserved_flow, score_utility
from reckless_uci import RecklessUci
from conductivity_positions import DEFAULT_REVISION, FenSource, board_position, board_state, restore_board


@dataclass
class Node:
    board: chess.Board
    parent: int = -1
    move: str = ''
    depth: int = 0
    value: float = 0.
    initial: float = 0.
    children: list = field(default_factory=list)
    deposit: float = 0.
    visits: int = 0


def terminal_value(board, root_turn):
    # Automatic draws only, not optional claims. Board retains path history.
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    return 0. if outcome.winner is None else (1. if outcome.winner == root_turn else -1.)


def reference_values(engine, board, depth):
    moves = engine.legal_moves(board.fen())
    root_fen, history = board_position(board)
    result = engine.analyze(root_fen, depth, multipv=len(moves), moves=history)
    infos = {i.pv[0]: i for i in result.infos if i.pv and i.bound is None and i.depth >= depth}
    values = {move: score_utility(info) for move, info in infos.items()}
    # Reckless can leave a bound-only MultiPV entry at the final depth.
    # Never train against it as if it were an exact evaluation. Re-search that
    # child independently at the same root-equivalent depth, reversing POV.
    for move in moves:
        if move in values:
            continue
        child = board.copy(stack=True)
        child.push_uci(move)
        terminal = terminal_value(child, board.turn)
        if terminal is not None:
            values[move] = terminal
            continue
        child_root, child_history = board_position(child)
        retry = engine.analyze(child_root, depth - 1, moves=child_history)
        exact = [i for i in retry.infos if i.bound is None and i.depth >= depth - 1]
        if not exact:
            raise RuntimeError(f'no completed exact reference for {move}')
        values[move] = -score_utility(exact[0])
    return values


def search(head, engine, board, budget=256, max_depth=4, stochastic=True, qnodes=4096):
    """Budget counts frontier evaluations, not internal quiescence nodes.

    Every expansion evaluates ALL children; never hides unsearched siblings.
    Quiescence node usage is reported separately; tactical extensions exceed the
    explicit tree depth cap. This is not an equal-wall-time claim.
    """
    root_turn = board.turn
    nodes = [Node(board.copy(stack=True))]
    frontier, log_probs = set(), []
    used, steps, qsearch_nodes, truncated_evaluations = 0, 0, 0, 0
    device = next(head.parameters()).device
    engine.new_game()  # Independent trajectories cannot share TT/history evidence.

    def expand(index):
        nonlocal used, qsearch_nodes, truncated_evaluations
        node = nodes[index]
        moves = engine.legal_moves(node.board.fen())
        if len(moves) > budget - used:
            return False
        boards = []
        for move in moves:
            child = node.board.copy(stack=True)
            child.push_uci(move)
            boards.append(child)
        terminals = [terminal_value(b, root_turn) for b in boards]
        scores = iter(engine.quiescence_evaluate([board_position(b) for b, terminal in zip(boards, terminals) if terminal is None], max_nodes=qnodes))
        used += len(boards)
        for move, child, terminal in zip(moves, boards, terminals):
            if terminal is None:
                info = next(scores)
                qsearch_nodes += info.nodes
                truncated_evaluations += info.truncated
                value = score_utility(info) * (1 if child.turn == root_turn else -1)
            else:
                value = terminal
            i = len(nodes)
            nodes.append(Node(child, index, move, node.depth + 1, value, value))
            node.children.append(i)
            if terminal is None and nodes[i].depth < max_depth:
                frontier.add(i)
        frontier.discard(index)
        while index >= 0:
            parent = nodes[index]
            if parent.children:
                parent.value = (max if parent.board.turn == root_turn else min)(nodes[c].value for c in parent.children)
            index = parent.parent
        return True

    if terminal_value(board, root_turn) is not None:
        raise ValueError('search requires nonterminal root')
    if not expand(0):
        raise ValueError('budget must cover all root children')
    while frontier and used < budget:
        conduct = {}
        for node in nodes:
            if not node.children:
                continue
            sign = 1 if node.board.turn == root_turn else -1
            xs = []
            for c in node.children:
                child = nodes[c]
                xs.append(features(node.board, chess.Move.from_uci(child.move), [
                    child.depth / max_depth, used / budget, sign * child.initial,
                    sign * child.value, sign * node.value, child.value - child.initial,
                    math.log1p(child.visits) / 10., float(c in frontier),
                    float(bool(child.children)), child.deposit,
                ]))
            # Progressive re-inference is intentional: fresh observed branch evidence
            # conditions the local prior; transported evidence remains separate.
            ds = head(torch.tensor(xs, dtype=torch.float32, device=device))
            for c, d in zip(node.children, ds):
                conduct[c] = d + nodes[c].deposit
        flow = conserved_flow([n.children for n in nodes], conduct, frontier)
        candidates = sorted(flow)
        distribution = Categorical(probs=torch.stack([flow[c] for c in candidates]))
        # CPU sampler gives checkpoints one portable, restorable RNG stream.
        action = torch.multinomial(distribution.probs.detach().cpu(), 1)[0].to(device) if stochastic else distribution.probs.argmax()
        index = candidates[action.item()]
        log_probs.append(distribution.log_prob(action))
        old_value = nodes[0].value
        if not expand(index):
            # Oversized branches close for this budget, so other smaller expansions
            # can still use the remaining compute. This choice is part of trajectory.
            frontier.discard(index)
        else:
            utility = min(1., .05 + abs(nodes[0].value - old_value)
                          + abs(nodes[index].value - nodes[index].initial))
            for node in nodes:
                node.deposit *= .97
            c = index
            while c > 0:
                nodes[c].visits += 1
                nodes[c].deposit += .75 * utility / nodes[index].depth
                c = nodes[c].parent
            # Sampling proportional to flow makes this an unbiased single-sink
            # estimate of transported utility. Evidence memory is observed state;
            # score-function gradients include the probability of the whole path.
        steps += 1
    chosen = max(nodes[0].children, key=lambda c: nodes[c].value)
    log_probability = sum(log_probs) if log_probs else next(head.parameters()).sum() * 0.
    return nodes[chosen].move, log_probability, {'frontier_evaluations': used, 'qsearch_nodes': qsearch_nodes, 'qsearch_truncated': truncated_evaluations, 'flow_steps': steps, 'tree_nodes': len(nodes)}


def regret_loss(regrets, log_probs):
    """REINFORCE with independent leave-one-out baseline, no critic/auxiliary loss."""
    if len(regrets) < 2:
        raise ValueError('at least two independent trajectories are required')
    costs = log_probs[0].new_tensor(regrets)
    if (costs.max() - costs.min()).item() == 0:
        return torch.stack(log_probs).sum() * 0.
    baseline = (costs.sum() - costs) / (len(costs) - 1)
    return ((costs - baseline).detach() * torch.stack(log_probs)).mean()


def save_checkpoint(path, head, optimizer, update, args, rng, episode=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': head.state_dict(), 'optimizer': optimizer.state_dict(),
                'width': head.width, 'feature_version': FEATURE_VERSION, 'update': update,
                'config': vars(args), 'episode': episode, 'python_rng': rng.getstate(), 'torch_rng': torch.get_rng_state()}, temporary)
    temporary.replace(path)


def write_tensorboard_update(writer, row, update, evaluate_only):
    """Log metrics with stable train/eval prefixes and one global update step."""
    prefix = 'eval' if evaluate_only else 'train'
    for field in ('mean_regret', 'regret_std', 'optimal_fraction', 'gradient_norm',
                  'seconds', 'reference_seconds', 'search_seconds', 'optimizer_seconds'):
        writer.add_scalar(f'{prefix}/{field}', row[field], update)
    for field in ('frontier_evaluations', 'qsearch_nodes', 'qsearch_truncated',
                  'qsearch_truncation_rate', 'flow_steps', 'tree_nodes'):
        writer.add_scalar(f'{prefix}/search/{field}', row[field], update)
    writer.add_scalar(f'{prefix}/game/ply', row['game_ply'], update)
    writer.add_scalar(f'{prefix}/game/completed', int('game_result' in row), update)
    if 'game_result' in row:
        # White-perspective outcome is explicit; it is not the training reward.
        result = {'1-0': 1, '1/2-1/2': 0, '0-1': -1}[row['game_result']]
        writer.add_scalar(f'{prefix}/game/result_white', result, update)
        writer.add_scalar(f'{prefix}/game/length_plies', row['game_ply'], update)
    writer.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True, help='native build WITHOUT cs-search or physarum-search')
    parser.add_argument('--output', type=Path, default=Path('outputs/conductivity'))
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--qnodes', type=int, default=4096, help='maximum native quiescence nodes per frontier child')
    parser.add_argument('--reference-depth', type=int, default=12)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--positions', type=Path, help='optional JSON list with fen fields, for controlled tests')
    parser.add_argument('--dataset-split', default='strong+mid+low+early')
    parser.add_argument('--dataset-revision', default=DEFAULT_REVISION, help='Hub commit SHA for reproducible stream resume')
    parser.add_argument('--shuffle-buffer', type=int, default=1024)
    parser.add_argument('--evaluate-only', action='store_true', help='frozen held-out evaluation; use a different seed')
    args = parser.parse_args()
    if args.reference_depth <= args.max_depth or args.max_depth < 2:
        parser.error('require reference-depth > max-depth >= 2')
    if min(args.updates, args.budget, args.width, args.qnodes) < 1 or args.rollouts < 2:
        parser.error('positive counts and rollouts >= 2 required')
    if args.shuffle_buffer < 1:
        parser.error('--shuffle-buffer must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    head = ConductivityHead(args.width).to(args.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    start = 0
    episode = {}
    if args.resume:
        # Only load trusted local checkpoints (contains Python RNG/config objects).
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['feature_version'] != FEATURE_VERSION or state['width'] != args.width:
            raise ValueError('checkpoint architecture mismatch')
        if not args.evaluate_only:
            for key in ('seed', 'dataset_split', 'dataset_revision', 'shuffle_buffer', 'positions', 'budget', 'max_depth', 'qnodes', 'reference_depth', 'rollouts'):
                if state['config'].get(key) != vars(args).get(key):
                    raise ValueError(f'resume changed {key}; start a new run or restore the original setting')
        head.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['update']
        rng.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        episode = state.get('episode') or {}
    if args.evaluate_only:
        episode = {}
    source = FenSource(args, episode.get('consumed', 0))
    board = restore_board(episode.get('board'))
    games = episode.get('games', 0)

    def episode_state():
        return {'consumed': source.consumed, 'board': board_state(board), 'games': games}
    args.output.mkdir(parents=True, exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.output / 'tensorboard'))
    print(json.dumps({'event': 'training_started', 'device': args.device,
                      'parameters': sum(p.numel() for p in head.parameters()), 'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}), flush=True)
    completed = start
    with RecklessUci(args.engine, args.timeout) as engine, RecklessUci(args.engine, args.timeout) as reference:
        try:
            engine._send('uci')
            options = engine._read_until(lambda line: line == 'uciok')
            if any('option name CS' in line or 'option name Physarum' in line for line in options):
                raise ValueError('reference requires native Reckless, not a CS/Physarum build')
            if args.evaluate_only:
                start = 0
                rng.seed(args.seed)
                torch.manual_seed(args.seed)
            for update in range(start + 1, args.updates + 1):
                if board is None:
                    try:
                        board = source.next_board()
                    except StopIteration:
                        print(json.dumps({'event': 'dataset_exhausted', 'games': games}), flush=True)
                        break
                    print(json.dumps({'event': 'game_started', 'game': games + 1, 'fen': board.fen()}), flush=True)
                begun = time.monotonic()
                print(json.dumps({'event': 'reference_started', 'update': update, 'fen': board.fen()}), flush=True)
                reference_started = time.monotonic()
                values = reference_values(reference, board, args.reference_depth)
                reference_seconds = time.monotonic() - reference_started
                regrets, logs, counts, selected_moves = [], [], [], []
                search_started = time.monotonic()
                with torch.set_grad_enabled(not args.evaluate_only):
                    for _ in range(args.rollouts):
                        move, logp, stats = search(head, engine, board, args.budget, args.max_depth, qnodes=args.qnodes)
                        regrets.append(max(values.values()) - values[move])
                        logs.append(logp)
                        counts.append(stats)
                        selected_moves.append(move)
                search_seconds = time.monotonic() - search_started
                loss = regret_loss(regrets, logs)
                norm = torch.tensor(0.)
                optimizer_started = time.monotonic()
                if not args.evaluate_only:
                    optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise RuntimeError('non-finite conductivity gradient')
                    optimizer.step()
                optimizer_seconds = time.monotonic() - optimizer_started
                completed = update
                evaluated = sum(item['frontier_evaluations'] for item in counts)
                truncated = sum(item['qsearch_truncated'] for item in counts)
                row = {'event': 'update_completed', 'update': update, 'mean_regret': sum(regrets)/len(regrets),
                       'regret_std': statistics.pstdev(regrets),
                       'optimal_fraction': sum(regret <= 1e-8 for regret in regrets) / len(regrets),
                       'regrets': regrets, 'surrogate_loss': loss.item(), 'gradient_norm': norm.item(),
                       'seconds': time.monotonic()-begun, 'reference_seconds': reference_seconds,
                       'search_seconds': search_seconds, 'optimizer_seconds': optimizer_seconds,
                       'frontier_evaluations': evaluated / len(counts),
                       'qsearch_nodes': sum(item['qsearch_nodes'] for item in counts) / len(counts),
                       'qsearch_truncated': truncated / len(counts),
                       'qsearch_truncation_rate': truncated / evaluated if evaluated else 0.,
                       'flow_steps': sum(item['flow_steps'] for item in counts) / len(counts),
                       'tree_nodes': sum(item['tree_nodes'] for item in counts) / len(counts),
                       'searches': counts}
                # First independent search drives self-play for BOTH colors;
                # never pick the continuation by its privileged reference score.
                played = selected_moves[0]
                board.push_uci(played)
                row.update(played_move=played, game=games + 1, game_ply=len(board.move_stack))
                outcome = board.outcome(claim_draw=False)
                if outcome is not None:
                    games += 1
                    row.update(game_result=outcome.result(), termination=outcome.termination.name)
                    board = None
                print(json.dumps(row), flush=True)
                with (args.output / 'metrics.jsonl').open('a') as file:
                    file.write(json.dumps(row) + '\n')
                write_tensorboard_update(writer, row, update, args.evaluate_only)
                if not args.evaluate_only:
                    save_checkpoint(args.output / 'latest.pt', head, optimizer, completed, args, rng, episode_state())
        except KeyboardInterrupt:
            print('Interrupted; saving current weights. In-flight rollout is discarded.', flush=True)
        finally:
            if not args.evaluate_only:
                save_checkpoint(args.output / 'latest.pt', head, optimizer, completed, args, rng, episode_state())
            writer.close()
            source.close()


if __name__ == '__main__':
    main()
