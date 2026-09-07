"""Verify resident Burn CS actor logits and decisions against PyTorch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import torch

from burn_client import BurnCsInference, pytorch_logits
from controller_model import ControllerConfig, MaskedActorCritic
from controller_state import BranchView, make_observation


def observation(rng: random.Random, index: int):
    count = rng.randint(2, 48)
    branches = []
    for move_index in range(count):
        source = move_index % 64
        target = (move_index * 17 + 9) % 64
        move = f"{'abcdefgh'[source % 8]}{source // 8 + 1}{'abcdefgh'[target % 8]}{target // 8 + 1}"
        searches = rng.randint(0, 5)
        branches.append(
            BranchView(
                move=move,
                score=rng.randint(-20_000, 20_000),
                depth=searches,
                cumulative_nodes=rng.randint(0, 2_000_000),
                last_nodes=rng.randint(0, 100_000),
                cumulative_time_ms=rng.randint(0, 20_000),
                last_score_delta=rng.randint(-5_000, 5_000),
                bound=rng.choice((None, "exact", "lowerbound", "upperbound")),
                searched=searches > 0,
                legal=searches < 5 and move_index != index % count,
            )
        )
    initial_budget = rng.randint(8, 64)
    return make_observation(
        branches,
        rng.randint(0, initial_budget),
        initial_budget,
        5,
        rng.randint(0, 10_000_000),
        rng.randint(0, 100_000),
        rng.randint(0, initial_budget),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MaskedActorCritic(ControllerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    rng = random.Random(91)
    maximum_error = 0.0
    decisions = 0
    with BurnCsInference(args.binary, args.weights) as burn:
        for index in range(args.observations):
            item = observation(rng, index)
            expected = list(pytorch_logits(model, item))
            actual = burn.logits(item)
            finite = [
                abs(left - right)
                for left, right in zip(expected, actual, strict=True)
                if math.isfinite(left) and left > -1e30
            ]
            maximum_error = max(maximum_error, max(finite, default=0.0))
            decisions += max(range(len(expected)), key=expected.__getitem__) == max(
                range(len(actual)), key=actual.__getitem__
            )
    result = {
        "observations": args.observations,
        "maximum_absolute_logit_error": maximum_error,
        "identical_decisions": decisions,
        "decision_agreement": decisions / args.observations,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
