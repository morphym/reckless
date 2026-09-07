"""Online CS reinforcement learning with live high-depth Reckless rewards.

No labelled action corpus is consumed. Root positions are generated online by
temperature-sampling the frozen Reckless NNUE scores of native legal moves. A
separate deterministic high-depth Reckless search produces numerical reference
values for every legal root move. The lower-cap CS controller receives only
its own epistemic state and learns from telescoping reference-regret rewards.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch

from controller_model import ControllerConfig, MaskedActorCritic, batch_observations
from live_env import LiveCsEnv
from ppo import PpoConfig, Transition, assign_gae, ppo_update
from reckless_uci import RecklessUci
from train_controller import choose_device


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def run_signature(args: argparse.Namespace) -> dict[str, object]:
    """Configuration that must remain stable when resuming a run."""
    return {
        "reference_depth": args.reference_depth,
        "cs_depths": list(args.cs_depths),
        "budget_range": [args.minimum_budget, args.maximum_budget],
        "root_ply_range": [args.root_minimum_plies, args.root_maximum_plies],
        "root_temperature": [args.root_temperature_start, args.root_temperature_end],
        "controller_temperature": [args.controller_temperature_start, args.controller_temperature_end],
        "episodes_per_update": args.episodes_per_update,
    }


def parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(int(item) for item in value.split(","))
    if not depths or depths[0] <= 0 or tuple(sorted(set(depths))) != depths:
        raise argparse.ArgumentTypeError("depths must be strictly increasing positive integers")
    return depths


def anneal(start: float, end: float, progress: float) -> float:
    if start <= 0 or end <= 0:
        raise ValueError("temperatures must be positive")
    progress = min(max(progress, 0.0), 1.0)
    return start * (end / start) ** progress


def boltzmann_choice(items: tuple[str, ...], scores: list[int], temperature_cp: float, rng: random.Random) -> str:
    """Sample an unusual move from deterministic NNUE numbers."""
    if not items or len(items) != len(scores):
        raise ValueError("items and scores must be nonempty and aligned")
    if temperature_cp <= 0:
        raise ValueError("temperature_cp must be positive")
    maximum = max(scores)
    weights = [math.exp(max((score - maximum) / temperature_cp, -60.0)) for score in scores]
    return rng.choices(items, weights=weights, k=1)[0]


def generate_online_root(
    engine: RecklessUci,
    minimum_plies: int,
    maximum_plies: int,
    temperature_cp: float,
    rng: random.Random,
) -> tuple[str, list[str]]:
    """Generate a root using native moves and high-temperature NNUE sampling."""
    if minimum_plies < 0 or maximum_plies < minimum_plies:
        raise ValueError("invalid root ply range")
    target = rng.randint(minimum_plies, maximum_plies)
    fen = START_FEN
    history: list[str] = []
    last_trainable = (fen, history.copy())
    for _ in range(target):
        moves = engine.legal_moves(fen)
        if len(moves) < 2:
            break
        last_trainable = (fen, history.copy())
        child_scores = engine.static_evaluate_after_moves(fen, moves)
        root_scores = [-score for score in child_scores]
        move = boltzmann_choice(moves, root_scores, temperature_cp, rng)
        fen = engine.fen_after_move(fen, move)
        history.append(move)

    moves = engine.legal_moves(fen)
    return (fen, history) if len(moves) >= 2 else last_trainable


def high_depth_reference(engine: RecklessUci, fen: str, depth: int) -> dict[str, int]:
    """Return numerical values for every legal root move, never action labels."""
    moves = engine.legal_moves(fen)
    if len(moves) < 2:
        raise ValueError("online RL roots need at least two legal moves")
    result = engine.analyze(fen, depth=depth, multipv=len(moves))
    reference = {info.pv[0]: info.score for info in result.infos if info.pv}
    if set(reference) != set(moves):
        raise RuntimeError(f"high-depth reference covered {len(reference)}/{len(moves)} legal moves")
    return reference


def create_online_environment(
    engine_path: Path,
    seed: int,
    root_minimum_plies: int,
    root_maximum_plies: int,
    root_temperature_cp: float,
    reference_depth: int,
    budget: int,
    cs_depths: tuple[int, ...],
    timeout: float,
) -> tuple[LiveCsEnv, dict[str, object]]:
    rng = random.Random(seed)
    with RecklessUci(engine_path, timeout_seconds=timeout) as reference_engine:
        fen, move_history = generate_online_root(
            reference_engine,
            root_minimum_plies,
            root_maximum_plies,
            root_temperature_cp,
            rng,
        )
        reference = high_depth_reference(reference_engine, fen, reference_depth)

    cs_engine = RecklessUci(engine_path, timeout_seconds=timeout)
    try:
        env = LiveCsEnv(cs_engine, fen, reference, budget, cs_depths)
    except BaseException:
        cs_engine.close()
        raise
    metadata = {
        "fen": fen,
        "generation_moves": move_history,
        "legal_moves": len(reference),
        "reference_depth": reference_depth,
        "reference_best_value": max(reference.values()),
        "budget": budget,
    }
    return env, metadata


def collect_live_episodes(
    model: MaskedActorCritic,
    envs: list[LiveCsEnv],
    device: torch.device,
    controller_temperature: float,
    workers: int,
) -> list[list[Transition]]:
    """Batch policy inference on GPU and native searches across CPU workers."""
    if controller_temperature <= 0:
        raise ValueError("controller_temperature must be positive")
    trajectories: list[list[Transition]] = [[] for _ in envs]
    active = list(range(len(envs)))
    model.eval()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while active:
            observations = [envs[index].observation() for index in active]
            batch = batch_observations(observations, device)
            with torch.no_grad():
                logits, values = model(batch)
                tempered_logits = logits / controller_temperature
                distribution = torch.distributions.Categorical(logits=tempered_logits)
                sampled = distribution.sample()
                log_probabilities = distribution.log_prob(sampled)
            sampled_rows = sampled.cpu().tolist()
            log_probability_rows = log_probabilities.cpu().tolist()
            value_rows = values.cpu().tolist()
            batch_stop_index = batch.candidates.shape[1]

            actions = []
            for row, env_index in enumerate(active):
                is_stop = sampled_rows[row] == batch_stop_index
                local_index = observations[row].stop_index if is_stop else sampled_rows[row]
                actions.append(envs[env_index].action_for_index(local_index))
            futures = [
                executor.submit(envs[env_index].step, action)
                for env_index, action in zip(active, actions, strict=True)
            ]
            results = [future.result() for future in futures]

            still_active = []
            for row, env_index in enumerate(active):
                is_stop = sampled_rows[row] == batch_stop_index
                trajectories[env_index].append(
                    Transition(
                        observation=observations[row],
                        action=-1 if is_stop else sampled_rows[row],
                        old_log_probability=log_probability_rows[row],
                        reward=float(results[row].reward),
                        value=value_rows[row],
                        temperature=controller_temperature,
                    )
                )
                if not envs[env_index].terminated:
                    still_active.append(env_index)
            active = still_active
    return trajectories


def write_tensorboard_update(
    writer,
    row: dict[str, object],
    initial_losses: list[int],
    terminal_losses: list[int],
    returns: list[float],
    node_counts: list[int],
) -> None:
    """Write one complete update without exposing privileged reference values."""
    step = int(row["update"])
    scalar_fields = {
        "ppo/policy_loss": "policy_loss",
        "ppo/value_loss": "value_loss",
        "ppo/entropy": "entropy",
        "quality/mean_initial_regret_cp": "mean_initial_loss",
        "quality/mean_terminal_regret_cp": "mean_terminal_loss",
        "quality/mean_regret_reduction_cp": "mean_return",
        "search/mean_nodes": "mean_nodes",
        "rollout/episodes": "episodes",
        "rollout/transitions": "transitions",
        "schedule/root_temperature_cp": "root_temperature_cp",
        "schedule/controller_temperature": "controller_temperature",
        "performance/update_seconds": "update_seconds",
        "performance/episodes_per_second": "episodes_per_second",
        "invariants/rewards_telescope": "all_rewards_telescope",
    }
    for tag, field in scalar_fields.items():
        writer.add_scalar(tag, float(row[field]), step)

    distributions = {
        "episode/initial_regret_cp": initial_losses,
        "episode/terminal_regret_cp": terminal_losses,
        "episode/regret_reduction_cp": returns,
        "episode/nodes": node_counts,
    }
    for tag, values in distributions.items():
        writer.add_histogram(tag, torch.tensor(values, dtype=torch.float32), step)
    writer.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=REPO_ROOT / "target/release/reckless")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/cs_online/controller.pt")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "outputs/cs_online/training.json")
    parser.add_argument("--tensorboard-dir", type=Path, default=REPO_ROOT / "outputs/cs_online/tensorboard")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--updates", type=int, default=1_000)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reference-depth", type=int, default=10)
    parser.add_argument("--cs-depths", type=parse_depths, default=(1, 2, 3, 4))
    parser.add_argument("--minimum-budget", type=int, default=4)
    parser.add_argument("--maximum-budget", type=int, default=32)
    parser.add_argument("--root-minimum-plies", type=int, default=8)
    parser.add_argument("--root-maximum-plies", type=int, default=80)
    parser.add_argument("--root-temperature-start", type=float, default=400.0)
    parser.add_argument("--root-temperature-end", type=float, default=40.0)
    parser.add_argument("--controller-temperature-start", type=float, default=3.0)
    parser.add_argument("--controller-temperature-end", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.updates <= 0 or args.episodes_per_update <= 0 or args.workers <= 0:
        raise ValueError("updates, episodes-per-update, and workers must be positive")
    if args.reference_depth <= args.cs_depths[-1] + 1:
        raise ValueError("root reference depth must exceed the branch CS depth cap plus the root move")
    if args.minimum_budget <= 0 or args.maximum_budget < args.minimum_budget:
        raise ValueError("invalid budget range")

    device = choose_device(args.device)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = MaskedActorCritic(ControllerConfig()).to(device)
    ppo_config = PpoConfig()
    optimizer = torch.optim.AdamW(model.parameters(), lr=ppo_config.learning_rate)
    history = []
    start_update = 0
    prior_elapsed = 0.0
    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "run_signature" in saved and saved["run_signature"] != run_signature(args):
            raise ValueError("resume configuration does not match the checkpoint run signature")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_update = int(saved["completed_updates"])
        if "python_rng_state" in saved:
            rng.setstate(saved["python_rng_state"])
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and "cuda_rng_state_all" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if args.summary.exists():
            previous_summary = json.loads(args.summary.read_text())
            history = previous_summary.get("history", [])
            prior_elapsed = float(previous_summary.get("elapsed_seconds", 0.0))

    writer = None
    close_writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        purge_step = start_update + 1 if args.resume and start_update > 0 else None
        writer = SummaryWriter(log_dir=str(args.tensorboard_dir), purge_step=purge_step)
        close_writer = writer.close
        atexit.register(close_writer)
        if start_update == 0:
            writer.add_text(
                "run/configuration",
                "```json\n" + json.dumps(run_signature(args), indent=2) + "\n```",
                global_step=0,
            )
            writer.flush()
    started = time.perf_counter()

    for update in range(start_update + 1, args.updates + 1):
        update_started = time.perf_counter()
        progress = (update - 1) / max(args.updates - 1, 1)
        root_temperature = anneal(args.root_temperature_start, args.root_temperature_end, progress)
        controller_temperature = anneal(
            args.controller_temperature_start,
            args.controller_temperature_end,
            progress,
        )
        budgets = [rng.randint(args.minimum_budget, args.maximum_budget) for _ in range(args.episodes_per_update)]
        seeds = [rng.randrange(2**63) for _ in range(args.episodes_per_update)]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    create_online_environment,
                    args.engine,
                    seed,
                    args.root_minimum_plies,
                    args.root_maximum_plies,
                    root_temperature,
                    args.reference_depth,
                    budget,
                    args.cs_depths,
                    args.timeout,
                )
                for seed, budget in zip(seeds, budgets, strict=True)
            ]
            created = [future.result() for future in futures]
        envs = [item[0] for item in created]
        metadata = [item[1] for item in created]
        initial_losses = [env.loss for env in envs]
        try:
            trajectories = collect_live_episodes(
                model,
                envs,
                device,
                controller_temperature,
                args.workers,
            )
            transitions = []
            for trajectory in trajectories:
                assign_gae(trajectory, ppo_config.gae_lambda)
                transitions.extend(trajectory)
            metrics = ppo_update(model, optimizer, transitions, ppo_config, device, rng)
            returns = [sum(item.reward for item in trajectory) for trajectory in trajectories]
            terminal_losses = [env.loss for env in envs]
            node_counts = [env.total_nodes for env in envs]
            update_seconds = time.perf_counter() - update_started
            row = {
                "update": update,
                "episodes": len(envs),
                "transitions": len(transitions),
                "root_temperature_cp": root_temperature,
                "controller_temperature": controller_temperature,
                "mean_initial_loss": sum(initial_losses) / len(initial_losses),
                "mean_terminal_loss": sum(terminal_losses) / len(terminal_losses),
                "mean_return": sum(returns) / len(returns),
                "mean_nodes": sum(node_counts) / len(node_counts),
                "update_seconds": update_seconds,
                "episodes_per_second": len(envs) / update_seconds,
                "all_rewards_telescope": all(
                    math.isclose(return_, initial_loss - env.loss)
                    for env, initial_loss, return_ in zip(envs, initial_losses, returns, strict=True)
                ),
                "root_examples": metadata[:2] if update in (1, args.updates) else [],
                **metrics,
            }
            history.append(row)
            if writer is not None:
                write_tensorboard_update(
                    writer,
                    row,
                    initial_losses,
                    terminal_losses,
                    returns,
                    node_counts,
                )
            print(json.dumps(row), flush=True)
        finally:
            for env in envs:
                env.close()

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "completed_updates": update,
            "online_rl": True,
            "run_signature": run_signature(args),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if device.type == "cuda":
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.checkpoint)

        # Keep progress metadata recoverable alongside the every-update model
        # checkpoint. This file is overwritten, not accumulated as a corpus.
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "engine": str(args.engine),
                    "checkpoint": str(args.checkpoint),
                    "device": str(device),
                    "tensorboard_dir": None if args.no_tensorboard else str(args.tensorboard_dir),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "completed_updates": update,
                    "target_updates": args.updates,
                    "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
                    "history": history,
                    "incomplete": update < args.updates,
                },
                indent=2,
            )
            + "\n"
        )

    summary = {
        "engine": str(args.engine),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "tensorboard_dir": None if args.no_tensorboard else str(args.tensorboard_dir),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "configuration": {
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "workers": args.workers,
            "reference_depth": args.reference_depth,
            "cs_depths": list(args.cs_depths),
            "budget_range": [args.minimum_budget, args.maximum_budget],
            "root_ply_range": [args.root_minimum_plies, args.root_maximum_plies],
            "root_temperature_cp": [args.root_temperature_start, args.root_temperature_end],
            "controller_temperature": [args.controller_temperature_start, args.controller_temperature_end],
            "seed": args.seed,
        },
        "ppo": asdict(ppo_config),
        "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        "history": history,
        "semantics": {
            "precomputed_corpus": False,
            "reference_is_numerical_reward_only": True,
            "reference_actions_exposed": False,
            "reference_depth_exceeds_cs_cap": True,
            "reward": "L_t - L_t+1",
            "gamma": 1.0,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    if close_writer is not None:
        close_writer()
        atexit.unregister(close_writer)
    print(json.dumps({"checkpoint": str(args.checkpoint), "summary": str(args.summary)}, indent=2))


if __name__ == "__main__":
    main()
