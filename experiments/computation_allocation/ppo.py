"""Small dependency-light PPO implementation for the CS controller."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch

from controller_model import MaskedActorCritic, batch_observations
from controller_state import ControllerObservation


@dataclass
class Transition:
    observation: ControllerObservation
    action: int  # Candidate index, or -1 for STOP.
    old_log_probability: float
    reward: float
    value: float
    temperature: float = 1.0
    advantage: float = 0.0
    return_: float = 0.0


@dataclass(frozen=True)
class PpoConfig:
    learning_rate: float = 3.0e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.02
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatch_size: int = 128
    max_grad_norm: float = 1.0
    value_scale_cp: float = 1_000.0
    value_huber_delta: float = 1.0


def collect_episode(
    model: MaskedActorCritic,
    env,
    device: torch.device,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> list[Transition]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    trajectory: list[Transition] = []
    model.eval()
    while not env.terminated:
        observation = env.observation()
        batch = batch_observations([observation], device)
        with torch.no_grad():
            logits, value = model(batch)
            tempered_logits = logits / temperature
            probabilities = torch.softmax(tempered_logits[0], dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).item()
            distribution = torch.distributions.Categorical(logits=tempered_logits)
            action_tensor = torch.tensor([sampled], device=device)
            log_probability = distribution.log_prob(action_tensor).item()

        is_stop = sampled == observation.stop_index
        stored_action = -1 if is_stop else sampled
        action = env.action_for_index(sampled) if hasattr(env, "action_for_index") else (
            "STOP" if is_stop else observation.moves[sampled]
        )
        result = env.step(action)
        trajectory.append(
            Transition(
                observation=observation,
                action=stored_action,
                old_log_probability=log_probability,
                reward=float(result.reward),
                value=float(value.item()),
                temperature=temperature,
            )
        )

    return trajectory


def collect_episodes(
    model: MaskedActorCritic,
    envs: Sequence,
    device: torch.device,
    temperature: float = 1.0,
) -> list[list[Transition]]:
    """Collect several episodes with one accelerator launch per CS step."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    trajectories: list[list[Transition]] = [[] for _ in envs]
    active = list(range(len(envs)))
    model.eval()
    while active:
        observations = [envs[index].observation() for index in active]
        batch = batch_observations(observations, device)
        with torch.no_grad():
            logits, values = model(batch)
            tempered_logits = logits / temperature
            probabilities = torch.softmax(tempered_logits, dim=-1)
            sampled = torch.multinomial(probabilities, 1).squeeze(1)
            distribution = torch.distributions.Categorical(logits=tempered_logits)
            log_probabilities = distribution.log_prob(sampled)
        sampled_rows = sampled.cpu().tolist()
        log_probability_rows = log_probabilities.cpu().tolist()
        value_rows = values.cpu().tolist()
        batch_stop_index = batch.candidates.shape[1]

        still_active = []
        for row, env_index in enumerate(active):
            observation = observations[row]
            sampled_index = sampled_rows[row]
            is_stop = sampled_index == batch_stop_index
            stored_action = -1 if is_stop else sampled_index
            action = envs[env_index].action_for_index(
                observation.stop_index if is_stop else sampled_index
            )
            result = envs[env_index].step(action)
            trajectories[env_index].append(
                Transition(
                    observation=observation,
                    action=stored_action,
                    old_log_probability=log_probability_rows[row],
                    reward=float(result.reward),
                    value=value_rows[row],
                    temperature=temperature,
                )
            )
            if not envs[env_index].terminated:
                still_active.append(env_index)
        active = still_active
    return trajectories


def assign_gae(trajectory: list[Transition], gae_lambda: float, value_scale_cp: float = 1_000.0) -> None:
    """Compute GAE in scaled critic units while retaining raw rewards."""
    if value_scale_cp <= 0:
        raise ValueError("value_scale_cp must be positive")
    next_value = 0.0
    next_advantage = 0.0
    for transition in reversed(trajectory):
        scaled_reward = transition.reward / value_scale_cp
        delta = scaled_reward + next_value - transition.value
        transition.advantage = delta + gae_lambda * next_advantage
        transition.return_ = transition.advantage + transition.value
        next_value = transition.value
        next_advantage = transition.advantage


def ppo_update(
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    config: PpoConfig,
    device: torch.device,
    rng: random.Random,
) -> dict[str, float]:
    if not transitions:
        raise ValueError("PPO update needs at least one transition")
    if config.value_scale_cp <= 0 or config.value_huber_delta <= 0:
        raise ValueError("critic scale and Huber delta must be positive")
    advantages_all = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
    advantage_mean = advantages_all.mean()
    advantage_std = advantages_all.std(unbiased=False).clamp_min(1.0e-8)
    normalized = ((advantages_all - advantage_mean) / advantage_std).tolist()

    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "value_mae_cp": 0.0,
        "value_target_rms_cp": 0.0,
        "entropy": 0.0,
        "updates": 0.0,
    }
    indices = list(range(len(transitions)))
    model.train()
    for _ in range(config.epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            selected = indices[start : start + config.minibatch_size]
            items = [transitions[index] for index in selected]
            batch = batch_observations([item.observation for item in items], device)
            logits, values = model(batch)
            temperatures = torch.tensor(
                [item.temperature for item in items], dtype=torch.float32, device=device
            ).unsqueeze(1)
            distribution = torch.distributions.Categorical(logits=logits / temperatures)
            stop_index = batch.candidates.shape[1]
            actions = torch.tensor(
                [stop_index if item.action < 0 else item.action for item in items],
                dtype=torch.long,
                device=device,
            )
            old_log_probabilities = torch.tensor(
                [item.old_log_probability for item in items], dtype=torch.float32, device=device
            )
            returns = torch.tensor([item.return_ for item in items], dtype=torch.float32, device=device)
            advantages = torch.tensor([normalized[index] for index in selected], dtype=torch.float32, device=device)

            log_probabilities = distribution.log_prob(actions)
            ratios = (log_probabilities - old_log_probabilities).exp()
            unclipped = ratios * advantages
            clipped = ratios.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.smooth_l1_loss(
                values,
                returns,
                beta=config.value_huber_delta,
            )
            value_mae_cp = (values.detach() - returns).abs().mean() * config.value_scale_cp
            value_target_rms_cp = returns.square().mean().sqrt() * config.value_scale_cp
            entropy = distribution.entropy().mean()
            loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["value_mae_cp"] += float(value_mae_cp)
            totals["value_target_rms_cp"] += float(value_target_rms_cp)
            totals["entropy"] += float(entropy.detach())
            totals["updates"] += 1.0

    count = totals.pop("updates")
    return {name: value / count for name, value in totals.items()}


def deterministic_action(model: MaskedActorCritic, observation: ControllerObservation, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        logits, _ = model(batch_observations([observation], device))
    return int(logits[0].argmax().item())
