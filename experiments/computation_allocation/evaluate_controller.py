"""Compare a trained CS controller with equal-budget cached allocators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics

import torch

from allocation_env import STOP
from cached_env import CachedCsEnv
from controller_model import ControllerConfig, MaskedActorCritic
from ppo import deterministic_action
from train_controller import load_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("local/cs-controller-native-1k.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/cs_controller/controller-1k.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cs_controller/evaluation-1k.json"))
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def summarize(rows: list[dict]) -> dict[str, object]:
    deltas = [row["terminal_loss"] - row["initial_loss"] for row in rows]
    non_mate = [row for row in rows if not row["has_mate_reference"]]
    return {
        "episodes": len(rows),
        "mean_initial_loss": sum(row["initial_loss"] for row in rows) / len(rows),
        "mean_terminal_loss": sum(row["terminal_loss"] for row in rows) / len(rows),
        "median_loss_change": statistics.median(deltas),
        "improved": sum(delta < 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "worse": sum(delta > 0 for delta in deltas),
        "mean_steps": sum(row["steps"] for row in rows) / len(rows),
        "mean_nodes": sum(row["nodes"] for row in rows) / len(rows),
        "non_mate_episodes": len(non_mate),
        "non_mate_mean_initial_loss": sum(row["initial_loss"] for row in non_mate) / len(non_mate),
        "non_mate_mean_terminal_loss": sum(row["terminal_loss"] for row in non_mate) / len(non_mate),
        "all_rewards_telescope": all(row["telescopes"] for row in rows),
    }


def rollout(model, device, position, depths, budget, strategy, rng) -> dict:
    env = CachedCsEnv.from_position(position, depths, budget)
    initial_loss = env.loss
    reference_depth = str(position["reference_depth"])
    has_mate_reference = any(
        abs(int(per_depth[reference_depth]["score"])) >= 90_000
        for per_depth in position["branches"].values()
    )
    reward_sum = 0
    steps = 0
    rr_index = 0
    while not env.terminated:
        moves = [action for action in env.legal_actions if action != STOP]
        if not moves:
            action = STOP
        elif strategy == "learned":
            action = env.action_for_index(deterministic_action(model, env.observation(), device))
        elif strategy == "current-best":
            action = env.selected_move if env.selected_move in moves else moves[0]
        elif strategy == "round-robin":
            action = moves[rr_index % len(moves)]
            rr_index += 1
        elif strategy == "random":
            action = rng.choice(moves)
        else:
            raise ValueError(strategy)
        result = env.step(action)
        reward_sum += result.reward
        steps += int(action != STOP)
    return {
        "position": position["name"],
        "budget": budget,
        "initial_loss": initial_loss,
        "terminal_loss": env.loss,
        "return": reward_sum,
        "steps": steps,
        "nodes": env.env.charged_nodes,
        "has_mate_reference": has_mate_reference,
        "telescopes": reward_sum == initial_loss - env.loss,
    }


def main() -> None:
    args = parse_args()
    payload, depths = load_payload(args.input)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MaskedActorCritic(ControllerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = torch.device("cpu")
    rng = random.Random(args.seed)
    budgets = range(args.min_budget, args.max_budget + 1)

    results = {}
    for split in ("validation", "test"):
        positions = [position for position in payload["positions"] if position.get("split") == split]
        if not positions:
            continue
        strategies = {}
        for strategy in ("learned", "current-best", "round-robin", "random"):
            rows = [
                rollout(model, device, position, depths, budget, strategy, rng)
                for position in positions
                for budget in budgets
            ]
            strategies[strategy] = {"summary": summarize(rows), "rows": rows}
        results[split] = strategies

    output = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "budgets": [args.min_budget, args.max_budget],
        "seed": args.seed,
        "splits": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    compact = {
        split: {strategy: values["summary"] for strategy, values in strategies.items()}
        for split, strategies in results.items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
