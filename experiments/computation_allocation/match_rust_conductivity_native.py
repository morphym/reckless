#!/usr/bin/env python3
"""Play color-swapped games between Rust learned Physarum and native Reckless."""
import argparse
import json
from pathlib import Path
import time

import chess
import chess.pgn

from reckless_uci import RecklessUci


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow-engine', type=Path, required=True)
    parser.add_argument('--native-engine', type=Path, required=True)
    parser.add_argument('--flow-weights', type=Path, required=True)
    parser.add_argument('--positions', type=Path, default=HERE / 'positions.json')
    parser.add_argument('--position', default='tactical-attack')
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--qnodes', type=int, default=4096)
    parser.add_argument('--learned', action='store_true', help='use policy conductivity; default is heuristic prior')
    parser.add_argument('--max-plies', type=int, default=100)
    parser.add_argument('--native-depth', type=int, default=0,
                        help='if set, run native at this depth and give Physarum that measured time')
    parser.add_argument('--timeout', type=float, default=60.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    positions = {row['name']: row for row in json.loads(args.positions.read_text())}
    if args.position not in positions:
        parser.error(f'unknown position: {args.position}')
    fen = positions[args.position]['fen']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    games = []
    with RecklessUci(args.flow_engine, args.timeout) as flow, RecklessUci(args.native_engine, args.timeout) as native:
        flow.set_option('PhysarumWeights', args.flow_weights.resolve())
        flow.set_option('PhysarumBudget', args.budget)
        flow.set_option('PhysarumMaxDepth', args.max_depth)
        flow.set_option('PhysarumQNodes', args.qnodes)
        flow.set_option('PhysarumLearned', str(args.learned).lower())
        for number, flow_color in enumerate((chess.WHITE, chess.BLACK), 1):
            board = chess.Board(fen)
            moves = []
            started = time.perf_counter()
            while board.outcome(claim_draw=False) is None and len(moves) < args.max_plies:
                if args.native_depth:
                    n_started = time.perf_counter()
                    native_result = native.analyze(board.fen(), args.native_depth)
                    native_ms = (time.perf_counter() - n_started) * 1000.
                    f_started = time.perf_counter()
                    flow_result = flow.analyze_limited(board.fen(), 'movetime', max(1, round(native_ms)))
                    flow_ms = (time.perf_counter() - f_started) * 1000.
                else:
                    f_started = time.perf_counter()
                    flow_result = flow.analyze_limited(board.fen(), 'nodes', 1_000_000_000)
                    flow_ms = (time.perf_counter() - f_started) * 1000.
                    n_started = time.perf_counter()
                    native_result = native.analyze_limited(board.fen(), 'movetime', max(1, round(flow_ms)))
                    native_ms = (time.perf_counter() - n_started) * 1000.
                chosen = flow_result.bestmove if board.turn == flow_color else native_result.bestmove
                move = chess.Move.from_uci(chosen)
                if move not in board.legal_moves:
                    raise RuntimeError(f'illegal move {chosen}')
                moves.append({'ply': len(moves) + 1, 'played_by': 'flow' if board.turn == flow_color else 'native',
                              'move': chosen, 'flow_move': flow_result.bestmove, 'native_move': native_result.bestmove,
                              'flow_ms': flow_ms, 'native_ms': native_ms})
                board.push(move)
                print(json.dumps({'game': number, **moves[-1]}), flush=True)
            outcome = board.outcome(claim_draw=False)
            result = outcome.result() if outcome else '*'
            games.append({'game': number, 'flow_color': 'white' if flow_color else 'black',
                          'result': result, 'termination': outcome.termination.name if outcome else 'MAX_PLIES',
                          'plies': moves, 'final_fen': board.fen(),
                          'elapsed_seconds': time.perf_counter() - started})
            print(json.dumps({'event': 'game_completed', 'game': number, 'result': result,
                              'termination': games[-1]['termination'], 'plies': len(moves)}), flush=True)
    artifact = {'protocol': {'flow_engine': str(args.flow_engine.resolve()),
                             'native_engine': str(args.native_engine.resolve()),
                             'flow_weights': str(args.flow_weights.resolve()),
                             'position': args.position, 'budget': args.budget,
                             'max_depth': args.max_depth, 'qnodes': args.qnodes}, 'games': games}
    args.output.write_text(json.dumps(artifact, indent=2) + '\n')
    with args.output.with_suffix('.pgn').open('w') as handle:
        for record in games:
            game = chess.pgn.Game()
            game.setup(chess.Board(record['final_fen']))
            game.headers['Result'] = record['result']
            game.headers['White'] = 'Learned Physarum' if record['flow_color'] == 'white' else 'Native Reckless'
            game.headers['Black'] = 'Native Reckless' if record['flow_color'] == 'white' else 'Learned Physarum'
            game.headers['FEN'] = fen
            game.headers['SetUp'] = '1'
            # Rebuild from the initial FEN, since setup(final) would reverse the game.
            game = chess.pgn.Game()
            game.setup(chess.Board(fen))
            game.headers.update({'Result': record['result'], 'White': 'Learned Physarum' if record['flow_color'] == 'white' else 'Native Reckless',
                                 'Black': 'Native Reckless' if record['flow_color'] == 'white' else 'Learned Physarum'})
            node = game
            for ply in record['plies']:
                node = node.add_variation(chess.Move.from_uci(ply['move']))
            print(game, file=handle, end='\n\n')


if __name__ == '__main__':
    main()
