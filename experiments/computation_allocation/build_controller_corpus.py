"""Build cached CS branch curves with native Reckless search operations.

Deep reference values come from the existing Reckless depth-6 ordering corpus.
For each sampled root, every branch gets its own fresh native Reckless session;
successive depth actions on that branch retain TT/history state.  This makes the
cached transition depend on the visible per-branch depth index rather than on
the arbitrary order in which other branches were explored.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import random
import time

import pyarrow.dataset as ds

from reckless_uci import RecklessUci, normalize_score


REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE: RecklessUci | None = None


def parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(int(item) for item in value.split(","))
    if not depths or depths[0] <= 0 or tuple(sorted(set(depths))) != depths:
        raise argparse.ArgumentTypeError("depths must be strictly increasing positive integers")
    return depths


def initialize_worker(engine: str, timeout: float) -> None:
    global _ENGINE
    _ENGINE = RecklessUci(Path(engine), timeout_seconds=timeout)
    atexit.register(_ENGINE.close)


def reference_score(entry: dict) -> int:
    mate = entry.get("mate_distance")
    return normalize_score("mate", int(mate)) if mate is not None else int(entry["score_cp"])


def build_position(row: dict, depths: tuple[int, ...]) -> dict:
    assert _ENGINE is not None
    fen = row["fen"]
    labels = {entry["move_uci"]: entry for entry in row["ordering"]}

    _ENGINE.new_game()
    native_moves = _ENGINE.legal_moves(fen)
    if set(native_moves) != set(labels):
        raise RuntimeError(f"native move mismatch at {fen}")
    static_child_scores = _ENGINE.static_evaluate_after_moves(fen, native_moves)
    branches = {
        move: {
            "0": {
                "score": -child_score,
                "nodes": 0,
                "time_ms": 0,
                "bound": None,
            }
        }
        for move, child_score in zip(native_moves, static_child_scores, strict=True)
    }

    for move in native_moves:
        # A branch owns its cache trajectory. Other branches cannot invisibly
        # perturb it in this offline Markov approximation.
        _ENGINE.new_game()
        for depth in depths:
            result = _ENGINE.analyze_branch(fen, move, depth)
            info = result.infos[0]
            branches[move][str(depth)] = {
                "score": -info.score,
                "nodes": info.nodes,
                "time_ms": info.time_ms,
                "bound": {"lowerbound": "upperbound", "upperbound": "lowerbound"}.get(
                    info.bound, info.bound
                ),
            }

        deep = labels[move]
        branches[move][str(row["label_depth"])] = {
            "score": reference_score(deep),
            "nodes": 0,
            "time_ms": 0,
            "bound": deep.get("bound"),
        }

    return {
        "id": int(row["id"]),
        "name": f"native-{row['id']}",
        "fen": fen,
        "phase": row["phase"],
        "split": row["split"],
        "legal_moves": list(native_moves),
        "allocation_depths": [0, *depths],
        "reference_depth": int(row["label_depth"]),
        "branches": branches,
    }


def sample_rows(input_path: Path, split: str, target: int, seed: int) -> list[dict]:
    parts = sorted(input_path.glob("*.parquet")) if input_path.is_dir() else [input_path]
    dataset = ds.dataset([str(path) for path in parts], format="parquet")
    split_filter = None if split == "all" else ds.field("split") == split
    fragments = list(dataset.get_fragments(filter=split_filter))
    rng = random.Random(seed)
    rng.shuffle(fragments)
    rows = []
    per_fragment = max(1, math.ceil(target / max(len(fragments), 1)))
    for fragment in fragments:
        fragment_rows = fragment.to_table(
            columns=["id", "fen", "phase", "split", "ordering", "label_depth"],
            filter=split_filter,
        ).to_pylist()
        rng.shuffle(fragment_rows)
        rows.extend(fragment_rows[: min(per_fragment, target - len(rows))])
        if len(rows) >= target:
            break
    if len(rows) < target:
        raise ValueError(f"requested {target} rows for split={split}, found {len(rows)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("local/reckless-ordering-native-d6"))
    parser.add_argument("--engine", type=Path, default=REPO_ROOT / "target/release/reckless")
    parser.add_argument("--output", type=Path, default=Path("local/cs-controller-native.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("local/cs-controller-native-manifest.json"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--target", type=int, default=1_000)
    parser.add_argument("--depths", type=parse_depths, default=(1, 2, 3, 4))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target <= 0 or args.workers <= 0:
        raise ValueError("target and workers must be positive")
    rows = sample_rows(args.input, args.split, args.target, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(str(args.engine.resolve()), args.timeout),
    ) as executor:
        built = list(executor.map(build_position, rows, [args.depths] * len(rows)))

    with args.output.open("w", encoding="utf-8") as handle:
        for row in built:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    phase_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for row in built:
        phase_counts[row["phase"]] = phase_counts.get(row["phase"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    manifest = {
        "rows": len(built),
        "input": str(args.input),
        "output": str(args.output),
        "engine": str(args.engine),
        "split": args.split,
        "allocation_depths": [0, *args.depths],
        "reference_depth": 6,
        "workers": args.workers,
        "seed": args.seed,
        "phase_counts": phase_counts,
        "split_counts": split_counts,
        "wall_seconds": time.perf_counter() - started,
        "native_components": ["move-generator", "MovePicker", "alpha-beta", "NNUE", "TT-per-branch"],
        "reference_is_reward_only": True,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
