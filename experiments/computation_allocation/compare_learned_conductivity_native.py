#!/usr/bin/env python3
"""Matched observed-wall-time comparison of a conductivity checkpoint to native Reckless.

The current learned search is a Python/UCI prototype. Its wall time includes
feature construction, flow, policy inference, and native qsearch. Native uses
the same Reckless binary with UCI `go movetime` set to the prototype's observed
time. A cached separate MultiPV analysis scores the chosen root moves.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import chess
import torch

from conductivity_policy import ConductivityHead, FEATURE_VERSION
from reckless_uci import RecklessUci
from train_conductivity import search


HERE = Path(__file__).resolve().parent


def load_head(path):
    # This is a user supplied checkpoint from a trusted bucket. Torch's legacy
    # checkpoint includes Path/config/RNG objects as well as model tensors.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('feature_version') != FEATURE_VERSION:
        raise ValueError('checkpoint feature version differs from this search')
    head = ConductivityHead(state['width'])
    head.load_state_dict(state['model'])
    head.eval()
    return head, state.get('update')


def reference_cache(path, positions):
    data = json.loads(path.read_text())
    protocol = data.get('protocol', {})
    if protocol.get('reference') != 'separate native Reckless MultiPV depth 16':
        raise ValueError('reference cache must be separate native MultiPV depth 16')
    cached_fens = {row['position']: row['fen'] for row in data.get('positions', [])}
    refs = data['references_cp']
    for item in positions:
        if cached_fens.get(item['name']) != item['fen']:
            raise ValueError(f"cached reference FEN differs for {item['name']}")
        if not refs.get(item['name']):
            raise ValueError(f"no reference moves for {item['name']}")
    return refs


def score_move(reference, move):
    if move not in reference:
        raise RuntimeError(f'candidate move {move} is not in the legal-move reference')
    best = max(reference.values())
    return {'move': move, 'reference_score_cp': reference[move],
            'regret_cp': best - reference[move],
            'reference_best': reference[move] == best}


def summarize(rows, name):
    decisions = [row[name] for row in rows]
    return {'mean_regret_cp': statistics.fmean(item['regret_cp'] for item in decisions),
            'median_regret_cp': statistics.median(item['regret_cp'] for item in decisions),
            'reference_best_fraction': statistics.fmean(item['reference_best'] for item in decisions),
            'mean_elapsed_ms': statistics.fmean(item['elapsed_ms'] for item in decisions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--positions', type=Path, default=HERE / 'matched_cost_positions.json')
    parser.add_argument('--reference-cache', type=Path, default=HERE / 'artifacts/physarum-native-matched-cost-refined.json')
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--qnodes', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--limit', type=int, help='evaluate only the first N positions for a smoke test')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.budget, args.max_depth, args.qnodes, args.repeats) < 1 or args.limit is not None and args.limit < 1:
        parser.error('budget, depth, qnodes, and repeats must be positive')
    torch.set_num_threads(1)
    head, training_update = load_head(args.checkpoint)
    positions = json.loads(args.positions.read_text())
    if args.limit is not None:
        positions = positions[:args.limit]
    references = reference_cache(args.reference_cache, positions)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with RecklessUci(args.engine, args.timeout) as flow_engine, RecklessUci(args.engine, args.timeout) as native_engine:
        native_engine.analyze_limited(positions[0]['fen'], 'movetime', 200)
        for index, position in enumerate(positions):
            fen = position['fen']
            board = chess.Board(fen)
            reference = references[position['name']]
            if set(flow_engine.legal_moves(fen)) != set(reference):
                raise RuntimeError(f"reference legal moves differ for {position['name']}")
            for repeat in range(args.repeats):
                torch.manual_seed(args.seed + index * args.repeats + repeat)
                started = time.perf_counter()
                with torch.no_grad():
                    move, _, flow_stats = search(head, flow_engine, board, args.budget, args.max_depth,
                                                 qnodes=args.qnodes)
                flow_ms = (time.perf_counter() - started) * 1000.
                native_limit_ms = max(1, round(flow_ms))
                started = time.perf_counter()
                native_result = native_engine.analyze_limited(fen, 'movetime', native_limit_ms)
                native_ms = (time.perf_counter() - started) * 1000.
                row = {'position': position['name'], 'fen': fen, 'repeat': repeat + 1,
                       'flow': {**score_move(reference, move), 'elapsed_ms': flow_ms, **flow_stats},
                       'native': {**score_move(reference, native_result.bestmove),
                                  'requested_movetime_ms': native_limit_ms, 'elapsed_ms': native_ms,
                                  'reported_depth': native_result.infos[0].depth,
                                  'reported_nodes': native_result.infos[0].nodes}}
                rows.append(row)
                print(json.dumps({'position': row['position'], 'repeat': row['repeat'],
                                  'flow_move': move, 'native_move': native_result.bestmove,
                                  'flow_regret_cp': row['flow']['regret_cp'],
                                  'native_regret_cp': row['native']['regret_cp'],
                                  'flow_ms': round(flow_ms), 'native_ms': round(native_ms)}), flush=True)
                # Save after every pair so a slow or interrupted suite retains evidence.
                artifact = {'protocol': {'generated_at': datetime.now(timezone.utc).isoformat(),
                                         'host': platform.platform(), 'python': platform.python_version(),
                                         'checkpoint': str(args.checkpoint.resolve()),
                                         'checkpoint_update': training_update,
                                         'engine': str(args.engine.resolve()),
                                         'reference_cache': str(args.reference_cache.resolve()),
                                         'reference': 'separate native Reckless MultiPV depth 16',
                                         'comparison': 'flow measured wall time versus native go movetime',
                                         'flow_budget': args.budget, 'flow_max_depth': args.max_depth,
                                         'qnodes': args.qnodes, 'seed': args.seed, 'repeats': args.repeats},
                            'rows': rows, 'summary': {name: summarize(rows, name) for name in ('flow', 'native')}}
                args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    print(json.dumps(artifact['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
