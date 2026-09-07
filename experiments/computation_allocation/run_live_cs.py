"""Run a live computation-allocation episode through native Reckless search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from allocation_env import STOP
from live_env import LiveCsEnv
from reckless_uci import RecklessUci


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(int(item) for item in value.split(","))
    if not depths or tuple(sorted(set(depths))) != depths or depths[0] <= 0:
        raise argparse.ArgumentTypeError("depths must be strictly increasing positive integers")
    return depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/pre-policy-results.json"))
    parser.add_argument("--engine", type=Path, default=REPO_ROOT / "target/release/reckless")
    parser.add_argument("--position-index", type=int, default=0)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--depths", type=parse_depths, default=(1, 2, 3, 4))
    parser.add_argument(
        "--strategy",
        choices=("controller", "current-best", "round-robin", "random"),
        default="round-robin",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/cs_controller/controller-smoke.pt"))
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--output", type=Path, default=Path("outputs/cs_controller/live-smoke.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    position = payload["positions"][args.position_index]
    reference_depth = str(position["reference_depth"])
    reference = {
        move: int(per_depth[reference_depth]["score"])
        for move, per_depth in position["branches"].items()
    }
    rng = random.Random(args.seed)

    model = None
    device = None
    if args.strategy == "controller":
        import torch

        from controller_model import ControllerConfig, MaskedActorCritic

        device = torch.device("cpu")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = MaskedActorCritic(ControllerConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

    with RecklessUci(args.engine) as engine:
        env = LiveCsEnv(engine, position["fen"], reference, args.budget, args.depths)
        initial_move = env.selected_move
        initial_loss = env.loss
        steps = []
        round_robin_index = 0
        while not env.terminated:
            moves = [action for action in env.legal_actions if action != STOP]
            if not moves:
                action = STOP
            elif args.strategy == "controller":
                assert model is not None and device is not None
                from ppo import deterministic_action

                action = env.action_for_index(deterministic_action(model, env.observation(), device))
            elif args.strategy == "current-best":
                action = env.selected_move if env.selected_move in moves else moves[0]
            elif args.strategy == "random":
                action = rng.choice(moves)
            else:
                action = moves[round_robin_index % len(moves)]
                round_robin_index += 1
            result = env.step(action)
            steps.append(result.__dict__)

        reward_sum = sum(step["reward"] for step in steps)
        result = {
            "position": position["name"],
            "fen": position["fen"],
            "engine": str(args.engine),
            "native_move_count": len(env.branches),
            "strategy": args.strategy,
            "checkpoint": str(args.checkpoint) if model is not None else None,
            "budget": args.budget,
            "depth_schedule": list(args.depths),
            "initial_move": initial_move,
            "initial_loss": initial_loss,
            "steps": steps,
            "terminal_move": env.selected_move,
            "terminal_loss": env.loss,
            "reward_sum": reward_sum,
            "rewards_telescope": reward_sum == initial_loss - env.loss,
            "charged_nodes": env.total_nodes,
            "charged_time_ms": env.total_time_ms,
            "decision_changes": env.decision_changes,
            "persistent_native_search": True,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
