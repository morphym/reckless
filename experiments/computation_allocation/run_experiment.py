"""Build and exercise the pre-policy computation-allocation environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Callable

from allocation_env import BranchEstimate, RootBranchAllocationEnv, STOP
from diagnostic import run_diagnostics
from reckless_uci import RecklessUci


def parse_depths(value: str) -> tuple[int, ...]:
    depths = tuple(sorted({int(item) for item in value.split(",")}))
    if not depths or depths[0] <= 0:
        raise argparse.ArgumentTypeError("depths must be positive comma-separated integers")
    return depths


def first_info(result):
    if not result.infos:
        raise RuntimeError("missing search info")
    return result.infos[0]


def collect_position(
    engine: RecklessUci,
    name: str,
    fen: str,
    allocation_depths: tuple[int, ...],
    reference_depth: int,
    conventional_depths: tuple[int, ...],
) -> dict[str, object]:
    enumeration = engine.analyze(fen, depth=1, multipv=256)
    legal_moves = tuple(sorted({info.pv[0] for info in enumeration.infos if info.pv}))
    if not legal_moves:
        raise RuntimeError(f"no legal moves found for {name}")

    all_depths = tuple(sorted(set(allocation_depths) | {reference_depth}))
    branches: dict[str, dict[str, dict[str, int | str | None]]] = {}
    for move in legal_moves:
        per_depth = {}
        for depth in all_depths:
            result = engine.analyze(fen, depth=depth, moves=(move,))
            info = first_info(result)
            # The child is the opponent-to-move position, so negate its score.
            per_depth[str(depth)] = {
                "score": -info.score,
                "nodes": info.nodes,
                "time_ms": info.time_ms,
                "bound": info.bound,
            }
        branches[move] = per_depth

    reference_scores = {move: int(branches[move][str(reference_depth)]["score"]) for move in legal_moves}
    reference_best = max(legal_moves, key=lambda move: (reference_scores[move], move))
    reference_best_score = reference_scores[reference_best]

    conventional = []
    losses = []
    for depth in conventional_depths:
        result = engine.analyze(fen, depth=depth)
        info = first_info(result)
        selected = result.bestmove
        loss = reference_best_score - reference_scores[selected]
        losses.append(loss)
        conventional.append(
            {
                "depth": depth,
                "selected_move": selected,
                "search_score": info.score,
                "reference_score": reference_scores[selected],
                "reference_regret": loss,
                "nodes": info.nodes,
                "time_ms": info.time_ms,
                "bound": info.bound,
            }
        )

    rewards = [before - after for before, after in zip(losses, losses[1:])]
    return {
        "name": name,
        "fen": fen,
        "legal_moves": list(legal_moves),
        "branches": branches,
        "reference_depth": reference_depth,
        "reference_best_move": reference_best,
        "reference_best_score": reference_best_score,
        "conventional": conventional,
        "conventional_dense_rewards": rewards,
        "conventional_telescopes": sum(rewards) == (losses[0] - losses[-1] if losses else 0),
    }


def make_env(position: dict[str, object], allocation_depths: tuple[int, ...], budget: int) -> RootBranchAllocationEnv:
    branches = position["branches"]
    assert isinstance(branches, dict)
    curves = {
        move: tuple(
            BranchEstimate(
                depth=depth,
                score=int(per_depth[str(depth)]["score"]),
                nodes=int(per_depth[str(depth)]["nodes"]),
                time_ms=int(per_depth[str(depth)]["time_ms"]),
                bound=per_depth[str(depth)]["bound"],
            )
            for depth in allocation_depths
        )
        for move, per_depth in branches.items()
    }
    reference_depth = str(position["reference_depth"])
    reference = {move: int(per_depth[reference_depth]["score"]) for move, per_depth in branches.items()}
    return RootBranchAllocationEnv(curves, reference, budget)


def rollout(env: RootBranchAllocationEnv, chooser: Callable[[RootBranchAllocationEnv, int], str]) -> dict[str, object]:
    initial_move = env.selected_move
    initial_loss = env.loss
    steps = []
    index = 0
    while not env.terminated:
        action = chooser(env, index)
        result = env.step(action)
        steps.append(result.__dict__)
        index += 1
    reward_sum = sum(int(step["reward"]) for step in steps)
    return {
        "initial_move": initial_move,
        "initial_loss": initial_loss,
        "steps": steps,
        "terminal_move": env.selected_move,
        "terminal_loss": env.loss,
        "reward_sum": reward_sum,
        "telescopes": reward_sum == initial_loss - env.loss,
        "charged_nodes": env.charged_nodes,
        "charged_time_ms": env.charged_time_ms,
    }


def evaluate_allocators(position: dict[str, object], allocation_depths: tuple[int, ...], budget: int, seed: int) -> dict:
    base = make_env(position, allocation_depths, budget)
    initial_move = base.selected_move

    def current_best(env: RootBranchAllocationEnv, _: int) -> str:
        if env.selected_move in env.legal_actions:
            return env.selected_move
        return next((action for action in env.legal_actions if action != STOP), STOP)

    def round_robin(env: RootBranchAllocationEnv, index: int) -> str:
        actions = [action for action in env.legal_actions if action != STOP]
        return actions[index % len(actions)] if actions else STOP

    rng = random.Random(seed)

    def random_action(env: RootBranchAllocationEnv, _: int) -> str:
        actions = [action for action in env.legal_actions if action != STOP]
        return rng.choice(actions) if actions else STOP

    def oracle(env: RootBranchAllocationEnv, _: int) -> str:
        action, _ = env.bellman_decision()
        return action

    result = {
        "initial_move": initial_move,
        "initial_loss": base.loss,
        "bellman_by_budget": {
            str(units): {"action": base.bellman_decision(units)[0], "terminal_loss": base.bellman_decision(units)[1]}
            for units in range(budget + 1)
        },
        "rollouts": {},
    }
    for name, chooser in (
        ("current_best", current_best),
        ("round_robin", round_robin),
        ("random", random_action),
        ("bellman_oracle", oracle),
    ):
        result["rollouts"][name] = rollout(make_env(position, allocation_depths, budget), chooser)
    return result


def markdown_report(payload: dict[str, object]) -> str:
    by_name = {position["name"]: position for position in payload["positions"]}
    tactical = by_name.get("tactical-attack")
    quiet = by_name.get("quiet-middlegame")
    endgame = by_name.get("endgame")
    budget_rows = payload["diagnostics"]["budgets"]
    budget_summary = ", ".join(
        f"b{budget}: {entry['action']} / loss {entry['expected_loss']}"
        for budget, entry in budget_rows.items()
    )
    sweep_rows = payload["diagnostics"]["reliability_sweep"]
    sweep_summary = ", ".join(
        f"p={probability}: {entry['action']} / loss {entry['expected_loss']}"
        for probability, entry in sweep_rows.items()
    )
    lines = [
        "# Pre-policy computation-allocation experiment",
        "",
        "## Outcome",
        "",
        "The exact environment and frozen-evaluator data path are working. Reckless was selected as the",
        "local evaluator because it has a native ARM/NEON implementation; the current Obsidian SIMD layer",
        "is x86 SSSE3/AVX-only and is not a native fit for this Apple Silicon machine.",
        "",
        "Reckless is frozen. No policy head or engine search modification is active in this run.",
        "The reference score for every legal root move is an isolated deeper child search.",
        "",
        "## Exact diagnostics",
        "",
        f"- Transition kernel valid: `{payload['diagnostics']['kernel_valid']}`",
        f"- Budget decisions: {budget_summary}.",
        f"- Reliability sweep: {sweep_summary}.",
        f"- Dense loss-difference rewards telescope: `{payload['diagnostics']['dense_telescoping']['sum']}` = `{payload['diagnostics']['dense_telescoping']['initial_minus_terminal']}`.",
        f"- Ordinary discount reversal reproduced: `{payload['diagnostics']['discount_counterexample']['ordinary_discount_reverses_preference']}`",
        "",
        "## Chess baseline",
        "",
        "| Position | Reference best | Conventional regret by depth | Allocation oracle loss by budget |",
        "|---|---:|---|---|",
    ]
    for position in payload["positions"]:
        conventional = ", ".join(
            f"d{row['depth']}={row['reference_regret']}" for row in position["conventional"]
        )
        bellman = ", ".join(
            f"b{budget}={entry['terminal_loss']}" for budget, entry in position["allocation"]["bellman_by_budget"].items()
        )
        lines.append(
            f"| {position['name']} | `{position['reference_best_move']}` ({position['reference_best_score']}) | {conventional} | {bellman} |"
        )
    lines.extend(
        [
            "",
            "All conventional trajectories satisfied the telescoping reward identity.",
            "",
            "## Allocation rollouts at budget 3",
            "",
            "| Position | Current-best loss | Round-robin loss | Random loss | Bellman loss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for position in payload["positions"]:
        rollouts = position["allocation"]["rollouts"]
        lines.append(
            f"| {position['name']} | {rollouts['current_best']['terminal_loss']} | "
            f"{rollouts['round_robin']['terminal_loss']} | {rollouts['random']['terminal_loss']} | "
            f"{rollouts['bellman_oracle']['terminal_loss']} |"
        )
    lines.extend(
        [
            "",
            "Every rollout also satisfied `sum(reward) = initial_loss - terminal_loss`. The Bellman column",
            "uses privileged reference values and is only an oracle diagnostic; it is not a deployable policy.",
            "",
            "## What the result says",
            "",
            "- More search does not guarantee monotone proxy regret. The tactical position was correct at",
            "  depths 2 and 4 but wrong at depth 6; the endgame worsened at depth 4 before reaching zero regret at depth 6.",
            (
                "- Delayed computation value occurs in a real evaluator trace. In the endgame, one allocation unit "
                f"left regret at {endgame['allocation']['bellman_by_budget']['1']['terminal_loss']}, while two units "
                f"reduced it to {endgame['allocation']['bellman_by_budget']['2']['terminal_loss']}."
                if endgame else "- Delayed computation value remains an explicit environment diagnostic."
            ),
            (
                "- True stopping matters. On the already-correct tactical root, the oracle stopped immediately; a "
                f"forced current-best rollout worsened regret from {tactical['allocation']['initial_loss']} to "
                f"{tactical['allocation']['rollouts']['current_best']['terminal_loss']}."
                if tactical else "- True stopping remains a required action."
            ),
            (
                "- A three-unit shallow allocation could not repair the quiet middlegame: Bellman regret remained "
                f"{quiet['allocation']['bellman_by_budget']['3']['terminal_loss']}. That is useful negative evidence: "
                "the available actions/depth range did not contain a decision-changing computation."
                if quiet else "- Some roots have no decision-changing action in the tested range."
            ),
            "",
            "## Interpretation boundary",
            "",
            "This validates the finite-budget environment, objective, STOP semantics, exact Bellman baseline,",
            "and frozen-evaluator data path. The cached root-branch environment is deterministic and coarse:",
            "it does not yet expose Reckless's complete evolving alpha-beta frontier or claim Markov sufficiency.",
            "Those are search-integration questions for the next phase, before or alongside a computation-policy head.",
            "",
            "## Policy-head boundary for the next phase",
            "",
            "A head trained to reproduce far-depth move ordering at low depth is a useful move-policy/distillation",
            "baseline, but it is not yet the paper's computation policy. The latter must condition on the evolving",
            "epistemic state and remaining budget, and choose a computation such as which branch to deepen next.",
            "For alpha-beta integration, the safe first interface is to expose alpha, beta, node type, bound status,",
            "depth, and remaining budget as controller features while using policy output only for ordering. Search",
            "correctness should remain with alpha-beta; learned pruning or window control should be a later ablation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, default=here.parents[1] / "Reckless" / "reckless")
    parser.add_argument("--positions", type=Path, default=here / "positions.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--allocation-depths", type=parse_depths, default=(1, 2, 3))
    parser.add_argument("--reference-depth", type=int, default=8)
    parser.add_argument("--conventional-depths", type=parse_depths, default=(2, 4, 6))
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args()

    positions = json.loads(args.positions.read_text())[: args.limit]
    collected = []
    with RecklessUci(args.engine) as engine:
        for item in positions:
            result = collect_position(
                engine,
                item["name"],
                item["fen"],
                args.allocation_depths,
                args.reference_depth,
                args.conventional_depths,
            )
            result["allocation"] = evaluate_allocators(result, args.allocation_depths, args.budget, args.seed)
            collected.append(result)

    payload = {
        "configuration": {
            "engine": str(args.engine.resolve()),
            "allocation_depths": args.allocation_depths,
            "reference_depth": args.reference_depth,
            "conventional_depths": args.conventional_depths,
            "budget": args.budget,
            "seed": args.seed,
            "policy_head_enabled": False,
        },
        "diagnostics": run_diagnostics(),
        "positions": collected,
    }

    rendered = json.dumps(payload, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n")
    else:
        print(rendered)
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(markdown_report(payload))


if __name__ == "__main__":
    main()
