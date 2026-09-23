#!/usr/bin/env python3
"""Compare native Rust learned Physarum and native Reckless at paired wall time."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from compare_learned_conductivity_native import reference_cache, score_move
from reckless_uci import RecklessUci


HERE = Path(__file__).resolve().parent


def summarize(rows, name):
    scores = [row[name]['regret_cp'] for row in rows]
    return {'mean_regret_cp': statistics.fmean(scores),
            'median_regret_cp': statistics.median(scores),
            'reference_best_fraction': statistics.fmean(value == 0 for value in scores),
            'mean_elapsed_ms': statistics.fmean(row[name]['elapsed_ms'] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-engine', type=Path, required=True)
    parser.add_argument('--native-engine', type=Path, required=True)
    parser.add_argument('--flow-weights', type=Path, help='optional exported conductivity model for the flow engine')
    parser.add_argument('--positions', type=Path, default=HERE / 'matched_cost_positions.json')
    parser.add_argument('--reference-cache', type=Path, default=HERE / 'artifacts/physarum-native-matched-cost-refined.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    positions = json.loads(args.positions.read_text())
    refs = reference_cache(args.reference_cache, positions)
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with RecklessUci(args.flow_engine, 120) as flow, RecklessUci(args.native_engine, 120) as native:
        if args.flow_weights:
            flow.set_option('PhysarumWeights', args.flow_weights.resolve())
        calibration_started = time.perf_counter()
        native.analyze_limited(positions[0]['fen'], 'movetime', 200)
        calibration_ms = (time.perf_counter() - calibration_started) * 1000
        # Reckless's UCI movetime subtracts a small internal safety margin.
        # Compensate it so the observed times, not only the requests, are paired.
        compensation_ms = max(0, round(200 - calibration_ms))
        for item in positions:
            for repeat in range(args.repeats):
                fen = item['fen']
                flow.set_option('PhysarumSeed', 2026 + repeat)
                started = time.perf_counter()
                f = flow.analyze_limited(fen, 'nodes', 1_000_000_000)
                flow_ms = (time.perf_counter() - started) * 1000
                requested_ms = max(1, round(flow_ms + compensation_ms))
                started = time.perf_counter()
                n = native.analyze_limited(fen, 'movetime', requested_ms)
                native_ms = (time.perf_counter() - started) * 1000
                row = {'position': item['name'], 'repeat': repeat + 1,
                       'flow': {**score_move(refs[item['name']], f.bestmove),
                                'elapsed_ms': flow_ms, 'reported_depth': f.infos[0].depth,
                                'reported_nodes': f.infos[0].nodes},
                       'native': {**score_move(refs[item['name']], n.bestmove),
                                  'elapsed_ms': native_ms, 'requested_movetime_ms': requested_ms,
                                  'reported_depth': n.infos[0].depth, 'reported_nodes': n.infos[0].nodes}}
                rows.append(row)
                artifact = {'protocol': {'generated_at': datetime.now(timezone.utc).isoformat(),
                                         'flow_engine': str(args.flow_engine.resolve()),
                                         'native_engine': str(args.native_engine.resolve()),
                                         'reference': 'separate native Reckless MultiPV depth 16',
                                         'movetime_compensation_ms': compensation_ms,
                                         'repeats': args.repeats,
                                         'comparison': 'native Rust learned Physarum measured wall time versus native go movetime'},
                            'rows': rows, 'summary': {name: summarize(rows, name) for name in ('flow', 'native')}}
                args.output.write_text(json.dumps(artifact, indent=2) + '\n')
                print(json.dumps({'position': item['name'], 'repeat': repeat + 1,
                    'flow_move': f.bestmove, 'native_move': n.bestmove,
                    'flow_regret_cp': row['flow']['regret_cp'], 'native_regret_cp': row['native']['regret_cp'],
                    'flow_ms': round(flow_ms), 'native_ms': round(native_ms)}), flush=True)


if __name__ == '__main__':
    main()
