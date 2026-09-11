#!/usr/bin/env python3
"""Matched-cost comparison of Physarum and native Reckless search.

Wall time is the primary hardware-cost comparison.  The node comparison is
secondary: native alpha-beta nodes and Physarum frontier-plus-quiescence nodes
are deliberately reported separately because they are not the same unit of
work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

from reckless_uci import RecklessUci, SearchResult


HERE = Path(__file__).resolve().parent


def parse_positive_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def final_info(result: SearchResult):
    if not result.infos:
        raise RuntimeError("search returned no final info")
    return result.infos[0]


def reference_values(engine: RecklessUci, fen: str, depth: int) -> dict[str, int]:
    moves = engine.legal_moves(fen)
    result = engine.analyze(fen, depth=depth, multipv=len(moves))
    values = {info.pv[0]: info.score for info in result.infos if info.pv}
    if set(values) != set(moves):
        raise RuntimeError(f"reference covered {len(values)}/{len(moves)} legal moves")
    return values


def run_candidate(
    engine: RecklessUci,
    fen: str,
    limit_kind: str,
    limit: int,
    reference: dict[str, int],
) -> dict[str, object]:
    started = time.perf_counter()
    result = engine.analyze_limited(fen, limit_kind, limit)
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    info = final_info(result)
    if result.bestmove not in reference:
        raise RuntimeError(f"candidate returned unreferenced move {result.bestmove}")
    best_value = max(reference.values())
    selected_value = reference[result.bestmove]
    best_moves = sorted(move for move, value in reference.items() if value == best_value)
    return {
        "bestmove": result.bestmove,
        "reference_value_cp": selected_value,
        "regret_cp": best_value - selected_value,
        "reference_best_moves": best_moves,
        "bestmove_agreement": result.bestmove in best_moves,
        "reported_score": {
            "kind": info.score_kind,
            "raw": info.score_raw,
            "normalized": info.score,
        },
        "reported_depth": info.depth,
        "reported_nodes": info.nodes,
        "reported_time_ms": info.time_ms,
        "observed_elapsed_ms": elapsed_ms,
    }


def summarize(rows: list[dict[str, object]], engine_name: str, limit_kind: str, limit: int) -> dict[str, object]:
    selected = [
        row["result"]
        for row in rows
        if row["engine"] == engine_name and row["limit_kind"] == limit_kind and row["limit"] == limit
    ]
    regrets = [float(item["regret_cp"]) for item in selected]
    return {
        "engine": engine_name,
        "limit_kind": limit_kind,
        "limit": limit,
        "positions": len(
            {
                row["position"]
                for row in rows
                if row["engine"] == engine_name and row["limit_kind"] == limit_kind and row["limit"] == limit
            }
        ),
        "samples": len(selected),
        "bestmove_agreement_rate": sum(bool(item["bestmove_agreement"]) for item in selected) / len(selected),
        "mean_regret_cp": statistics.fmean(regrets),
        "median_regret_cp": statistics.median(regrets),
        "maximum_regret_cp": max(regrets),
        "mean_reported_nodes": statistics.fmean(float(item["reported_nodes"]) for item in selected),
        "mean_reported_time_ms": statistics.fmean(float(item["reported_time_ms"]) for item in selected),
        "mean_observed_elapsed_ms": statistics.fmean(float(item["observed_elapsed_ms"]) for item in selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--physarum", type=Path, required=True)
    parser.add_argument("--positions", type=Path, default=HERE / "matched_cost_positions.json")
    parser.add_argument("--reference-depth", type=int, default=16)
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="reuse references_cp from an earlier artifact with the same positions and depth",
    )
    parser.add_argument("--movetimes", type=parse_positive_csv, default=(100, 500))
    parser.add_argument("--node-budgets", type=parse_positive_csv, default=(512, 2048))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    positions = json.loads(args.positions.read_text())
    rows: list[dict[str, object]] = []
    references: dict[str, dict[str, int]] = {}
    cached_fens: dict[str, str] = {}
    if args.reference_cache:
        cached = json.loads(args.reference_cache.read_text())
        expected = f"separate native Reckless MultiPV depth {args.reference_depth}"
        if cached.get("protocol", {}).get("reference") != expected:
            raise ValueError(f"reference cache was not generated with depth {args.reference_depth}")
        references = cached["references_cp"]
        cached_fens = {row["position"]: row["fen"] for row in cached.get("positions", [])}
    with (
        RecklessUci(args.native, timeout_seconds=args.timeout) as reference_engine,
        RecklessUci(args.native, timeout_seconds=args.timeout) as native_engine,
        RecklessUci(args.physarum, timeout_seconds=args.timeout) as physarum_engine,
    ):
        warmup_fen = positions[0]["fen"]
        native_engine.analyze_limited(warmup_fen, "movetime", 500)
        physarum_engine.analyze_limited(warmup_fen, "movetime", 500)
        for index, position in enumerate(positions, start=1):
            name = position["name"]
            fen = position["fen"]
            if name in references:
                if cached_fens.get(name) != fen:
                    raise RuntimeError(f"cached reference FEN for {name} does not match the suite")
                reference = references[name]
                legal_moves = set(reference_engine.legal_moves(fen))
                if set(reference) != legal_moves:
                    raise RuntimeError(f"cached reference for {name} does not match its legal moves")
                source = "reference reused"
            else:
                reference = reference_values(reference_engine, fen, args.reference_depth)
                references[name] = reference
                source = "reference generated"
            print(f"[{index}/{len(positions)}] {name}: {source} ({len(reference)} moves)", flush=True)
            for limit_kind, limits in (("movetime", args.movetimes), ("nodes", args.node_budgets)):
                for limit in limits:
                    for repeat in range(1, args.repeats + 1):
                        for engine_name, engine in (("native", native_engine), ("physarum", physarum_engine)):
                            result = run_candidate(engine, fen, limit_kind, limit, reference)
                            rows.append(
                                {
                                    "position": name,
                                    "fen": fen,
                                    "engine": engine_name,
                                    "limit_kind": limit_kind,
                                    "limit": limit,
                                    "repeat": repeat,
                                    "result": result,
                                }
                            )
                            print(
                                f"  {limit_kind}={limit:<5} repeat={repeat} {engine_name:<8} "
                                f"move={result['bestmove']} regret={result['regret_cp']:>5}cp "
                                f"nodes={result['reported_nodes']:>8} time={result['reported_time_ms']:>4}ms",
                                flush=True,
                            )

    summaries = [
        summarize(rows, engine, limit_kind, limit)
        for limit_kind, limits in (("movetime", args.movetimes), ("nodes", args.node_budgets))
        for limit in limits
        for engine in ("native", "physarum")
    ]
    artifact = {
        "protocol": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "primary_metric": "equal UCI movetime on one thread",
            "secondary_metric": "equal reported node limit; units differ by engine",
            "reference": f"separate native Reckless MultiPV depth {args.reference_depth}",
            "threads": 1,
            "hash_mb": 32,
            "positions": len(positions),
            "repeats": args.repeats,
            "warmup": "500 ms per candidate engine before measurement",
            "native_binary": str(args.native.resolve()),
            "physarum_binary": str(args.physarum.resolve()),
        },
        "summaries": summaries,
        "positions": rows,
        "references_cp": references,
    }
    encoded = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded)

    print("\nSummary", flush=True)
    for item in summaries:
        print(
            f"{item['limit_kind']}={item['limit']:<5} {item['engine']:<8} "
            f"agree={item['bestmove_agreement_rate']:.1%} "
            f"mean-regret={item['mean_regret_cp']:.1f}cp "
            f"median={item['median_regret_cp']:.1f}cp max={item['maximum_regret_cp']:.0f}cp",
            flush=True,
        )


if __name__ == "__main__":
    main()
