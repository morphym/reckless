"""Train the first budget-conditioned CS actor-critic with masked PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from cached_env import CachedCsEnv
from controller_model import ControllerConfig, MaskedActorCritic
from ppo import PpoConfig, assign_gae, collect_episodes, deterministic_action, ppo_update


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_payload(path: Path) -> tuple[dict, tuple[int, ...]]:
    if path.suffix == ".jsonl":
        positions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not positions:
            raise ValueError("input contains no positions")
        payload = {"positions": positions}
        depths = tuple(int(depth) for depth in positions[0]["allocation_depths"])
    else:
        payload = json.loads(path.read_text())
        depths = tuple(int(depth) for depth in payload["configuration"]["allocation_depths"])
    if not payload.get("positions"):
        raise ValueError("input contains no positions")
    return payload, depths


def make_env(position: dict, depths: tuple[int, ...], budget: int) -> CachedCsEnv:
    return CachedCsEnv.from_position(position, depths, budget)


def evaluate(
    model: MaskedActorCritic,
    positions: list[dict],
    depths: tuple[int, ...],
    budgets: tuple[int, ...],
    device: torch.device,
) -> dict[str, object]:
    rows = []
    for position in positions:
        for budget in budgets:
            env = make_env(position, depths, budget)
            initial_loss = env.loss
            reward_sum = 0
            steps = 0
            while not env.terminated:
                observation = env.observation()
                index = deterministic_action(model, observation, device)
                action = env.action_for_index(index)
                result = env.step(action)
                reward_sum += result.reward
                steps += 1
            rows.append(
                {
                    "position": position["name"],
                    "budget": budget,
                    "initial_loss": initial_loss,
                    "terminal_loss": env.loss,
                    "return": reward_sum,
                    "steps": steps,
                    "telescopes": reward_sum == initial_loss - env.loss,
                }
            )
    return {
        "episodes": len(rows),
        "mean_initial_loss": sum(row["initial_loss"] for row in rows) / len(rows),
        "mean_terminal_loss": sum(row["terminal_loss"] for row in rows) / len(rows),
        "mean_return": sum(row["return"] for row in rows) / len(rows),
        "all_rewards_telescope": all(row["telescopes"] for row in rows),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/pre-policy-results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/cs_controller/controller.pt"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/cs_controller/training.json"))
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=64)
    parser.add_argument("--min-budget", type=int, default=1)
    parser.add_argument("--max-budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--evaluation-roots", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0:
        raise ValueError("updates and episodes-per-update must be positive")
    if args.min_budget < 0 or args.max_budget < args.min_budget:
        raise ValueError("invalid budget range")

    payload, depths = load_payload(args.input)
    positions = payload["positions"]
    train_positions = [position for position in positions if position.get("split") == "train"]
    if not train_positions:
        train_positions = positions
    validation_positions = [position for position in positions if position.get("split") == "validation"]
    test_positions = [position for position in positions if position.get("split") == "test"]
    device = choose_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = MaskedActorCritic(ControllerConfig()).to(device)
    ppo_config = PpoConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ppo_config.learning_rate)
    history = []
    started = time.perf_counter()

    for update in range(1, args.updates + 1):
        transitions = []
        episode_returns = []
        episode_losses = []
        envs = []
        for _ in range(args.episodes_per_update):
            position = rng.choice(train_positions)
            budget = rng.randint(args.min_budget, args.max_budget)
            envs.append(make_env(position, depths, budget))
        for env, trajectory in zip(envs, collect_episodes(model, envs, device), strict=True):
            assign_gae(trajectory, ppo_config.gae_lambda)
            transitions.extend(trajectory)
            episode_returns.append(sum(item.reward for item in trajectory))
            episode_losses.append(env.loss)

        metrics = ppo_update(model, optimizer, transitions, ppo_config, device, rng)
        history.append(
            {
                "update": update,
                "transitions": len(transitions),
                "mean_return": sum(episode_returns) / len(episode_returns),
                "mean_terminal_loss": sum(episode_losses) / len(episode_losses),
                **metrics,
            }
        )
        if update == 1 or update == args.updates or update % max(args.updates // 10, 1) == 0:
            row = history[-1]
            print(
                f"update={update} transitions={row['transitions']} "
                f"return={row['mean_return']:.2f} loss={row['mean_terminal_loss']:.2f} "
                f"entropy={row['entropy']:.3f}",
                flush=True,
            )

    budgets = tuple(range(args.min_budget, args.max_budget + 1))
    evaluation_sets = {}
    for name, subset in (("validation", validation_positions), ("test", test_positions)):
        if subset:
            evaluation_sets[name] = evaluate(model, subset[: args.evaluation_roots], depths, budgets, device)
    if not evaluation_sets:
        evaluation_sets["development"] = evaluate(
            model, positions[: args.evaluation_roots], depths, budgets, device
        )
    elapsed = time.perf_counter() - started
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        "candidate_feature_version": 1,
        "global_feature_version": 1,
    }
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.checkpoint)
    summary = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "budget_range": [args.min_budget, args.max_budget],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": elapsed,
        "ppo": asdict(ppo_config),
        "history": history,
        "position_counts": {
            "all": len(positions),
            "train": len(train_positions),
            "validation": len(validation_positions),
            "test": len(test_positions),
        },
        "evaluation": evaluation_sets,
        "warning": "This checkpoint validates the RL path; strength requires a larger corpus and held-out games.",
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("checkpoint", "device", "parameters", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
