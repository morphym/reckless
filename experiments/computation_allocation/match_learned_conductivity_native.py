#!/usr/bin/env python3
"""Play color-swapped games between the learned Physarum search and Reckless.

At every position, both searches analyze the same board. Native Reckless gets
UCI movetime equal to the measured wall time of the Python Physarum search.
Only the designated side's move is played. This keeps comparison time paired
even when the explicit tree's runtime changes drastically across positions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import chess
import chess.pgn
import torch

from compare_learned_conductivity_native import load_head
from reckless_uci import RecklessUci
from train_conductivity import search


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--positions', type=Path, default=HERE / 'matched_cost_positions.json')
    parser.add_argument('--position', default='start-position', help='name in the positions JSON')
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--qnodes', type=int, default=4096)
    parser.add_argument('--max-plies', type=int, default=120)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.budget, args.max_depth, args.qnodes, args.max_plies) < 1:
        parser.error('all budget and depth limits must be positive')
    positions = {item['name']: item for item in json.loads(args.positions.read_text())}
    if args.position not in positions:
        parser.error(f'unknown position: {args.position}')
    fen = positions[args.position]['fen']
    head, checkpoint_update = load_head(args.checkpoint)
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    games = []
    pgn_path = args.output.with_suffix('.pgn')
    with RecklessUci(args.engine, args.timeout) as flow_engine, RecklessUci(args.engine, args.timeout) as native_engine:
        native_engine.analyze_limited(fen, 'movetime', 200)
        for number, flow_color in enumerate((chess.WHITE, chess.BLACK), start=1):
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.headers['Event'] = 'Learned Physarum vs native Reckless, paired wall time'
            game.headers['Round'] = str(number)
            game.headers['White'] = 'Learned Physarum' if flow_color else 'Native Reckless'
            game.headers['Black'] = 'Native Reckless' if flow_color else 'Learned Physarum'
            game.headers['FEN'] = fen
            game.headers['SetUp'] = '1'
            node = game
            plies = []
            for ply in range(args.max_plies):
                outcome = board.outcome(claim_draw=False)
                if outcome is not None:
                    break
                torch.manual_seed(args.seed + number * 100000 + ply)
                started = time.perf_counter()
                with torch.no_grad():
                    flow_move, _, flow_stats = search(head, flow_engine, board,
                                                      args.budget, args.max_depth, qnodes=args.qnodes)
                flow_ms = (time.perf_counter() - started) * 1000.
                started = time.perf_counter()
                native = native_engine.analyze_limited(board.fen(), 'movetime', max(1, round(flow_ms)))
                native_ms = (time.perf_counter() - started) * 1000.
                chosen = flow_move if board.turn == flow_color else native.bestmove
                move = chess.Move.from_uci(chosen)
                if move not in board.legal_moves:
                    raise RuntimeError(f'illegal move {chosen} in game {number} ply {ply}')
                row = {'ply': ply + 1, 'side': 'white' if board.turn else 'black',
                       'played_by': 'flow' if board.turn == flow_color else 'native',
                       'played_move': chosen, 'flow_move': flow_move,
                       'native_move': native.bestmove, 'flow_ms': flow_ms,
                       'native_ms': native_ms, 'native_depth': native.infos[0].depth,
                       'native_nodes': native.infos[0].nodes, 'flow_stats': flow_stats}
                plies.append(row)
                node = node.add_variation(move)
                board.push(move)
                print(json.dumps({'game': number, **{key: row[key] for key in
                      ('ply', 'played_by', 'played_move', 'flow_ms', 'native_ms')}}), flush=True)
                outcome = board.outcome(claim_draw=False)
                if outcome is not None:
                    break
            outcome = board.outcome(claim_draw=False)
            result = outcome.result() if outcome else '*'
            game.headers['Result'] = result
            record = {'game': number, 'flow_color': 'white' if flow_color else 'black',
                      'start_fen': fen, 'result': result,
                      'termination': outcome.termination.name if outcome else 'MAX_PLIES_UNFINISHED',
                      'plies': plies, 'final_fen': board.fen()}
            games.append(record)
            artifact = {'protocol': {'generated_at': datetime.now(timezone.utc).isoformat(),
                                     'checkpoint': str(args.checkpoint.resolve()),
                                     'checkpoint_update': checkpoint_update,
                                     'engine': str(args.engine.resolve()),
                                     'position': args.position, 'budget': args.budget,
                                     'max_depth': args.max_depth, 'qnodes': args.qnodes,
                                     'max_plies': args.max_plies, 'seed': args.seed,
                                     'comparison': 'both analyze each position; native movetime equals observed flow time'},
                        'games': games}
            args.output.write_text(json.dumps(artifact, indent=2) + '\n')
            with pgn_path.open('w') as file:
                for played in games:
                    # Reconstruct PGN from recorded moves to avoid retaining
                    # a giant Python game object across many plies.
                    g = chess.pgn.Game()
                    g.setup(chess.Board(played['start_fen']))
                    g.headers['White'] = 'Learned Physarum' if played['flow_color'] == 'white' else 'Native Reckless'
                    g.headers['Black'] = 'Native Reckless' if played['flow_color'] == 'white' else 'Learned Physarum'
                    g.headers['Result'] = played['result']
                    g.headers['Round'] = str(played['game'])
                    current = g
                    for entry in played['plies']:
                        current = current.add_variation(chess.Move.from_uci(entry['played_move']))
                    print(g, file=file, end='\n\n')
            print(json.dumps({'event': 'game_completed', 'game': number, 'result': result,
                              'termination': record['termination'], 'plies': len(plies)}), flush=True)


if __name__ == '__main__':
    main()
