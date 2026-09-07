"""Variable-frontier masked actor-critic for CS allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from controller_state import CANDIDATE_FEATURES, GLOBAL_FEATURES, ControllerObservation


@dataclass(frozen=True)
class ControllerConfig:
    candidate_width: int = 128
    global_width: int = 96
    context_width: int = 192
    hidden_width: int = 128

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerBatch:
    candidates: Tensor
    global_features: Tensor
    candidate_mask: Tensor
    action_mask: Tensor


def batch_observations(
    observations: Sequence[ControllerObservation],
    device: torch.device | str,
) -> ControllerBatch:
    if not observations:
        raise ValueError("cannot batch zero observations")
    max_candidates = max(len(observation.moves) for observation in observations)
    batch_size = len(observations)
    candidate_dim = len(CANDIDATE_FEATURES)
    global_dim = len(GLOBAL_FEATURES)

    candidates = torch.zeros((batch_size, max_candidates, candidate_dim), dtype=torch.float32, device=device)
    globals_ = torch.zeros((batch_size, global_dim), dtype=torch.float32, device=device)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=device)
    action_mask = torch.zeros((batch_size, max_candidates + 1), dtype=torch.bool, device=device)

    for row, observation in enumerate(observations):
        count = len(observation.moves)
        candidates[row, :count] = torch.tensor(observation.candidates, dtype=torch.float32, device=device)
        globals_[row] = torch.tensor(observation.global_features, dtype=torch.float32, device=device)
        candidate_mask[row, :count] = True
        action_mask[row, :count] = torch.tensor(observation.action_mask[:-1], dtype=torch.bool, device=device)
        action_mask[row, max_candidates] = observation.action_mask[-1]

    return ControllerBatch(candidates, globals_, candidate_mask, action_mask)


class MaskedActorCritic(nn.Module):
    """Score legal computation candidates and a true STOP action."""

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        super().__init__()
        self.config = config
        self.candidate_encoder = nn.Sequential(
            nn.Linear(len(CANDIDATE_FEATURES), config.candidate_width),
            nn.SiLU(),
            nn.LayerNorm(config.candidate_width),
            nn.Linear(config.candidate_width, config.candidate_width),
            nn.SiLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURES), config.global_width),
            nn.SiLU(),
            nn.LayerNorm(config.global_width),
        )
        pooled_width = 2 * config.candidate_width + config.global_width
        self.context_encoder = nn.Sequential(
            nn.Linear(pooled_width, config.context_width),
            nn.SiLU(),
            nn.LayerNorm(config.context_width),
        )
        self.candidate_actor = nn.Sequential(
            nn.Linear(config.candidate_width + config.context_width, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, 1),
        )
        self.stop_actor = nn.Sequential(
            nn.Linear(config.context_width, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.context_width, config.hidden_width),
            nn.SiLU(),
            nn.Linear(config.hidden_width, 1),
        )

    def forward(self, batch: ControllerBatch) -> tuple[Tensor, Tensor]:
        encoded = self.candidate_encoder(batch.candidates)
        mask = batch.candidate_mask.unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1)
        mean_pool = (encoded * mask).sum(dim=1) / count
        max_pool = encoded.masked_fill(~mask, torch.finfo(encoded.dtype).min).max(dim=1).values
        global_encoded = self.global_encoder(batch.global_features)
        context = self.context_encoder(torch.cat((mean_pool, max_pool, global_encoded), dim=-1))

        repeated_context = context.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        candidate_logits = self.candidate_actor(torch.cat((encoded, repeated_context), dim=-1)).squeeze(-1)
        stop_logit = self.stop_actor(context)
        logits = torch.cat((candidate_logits, stop_logit), dim=-1)
        logits = logits.masked_fill(~batch.action_mask, torch.finfo(logits.dtype).min)
        value = self.critic(context).squeeze(-1)
        return logits, value

    def distribution(self, batch: ControllerBatch) -> tuple[Categorical, Tensor]:
        logits, value = self(batch)
        return Categorical(logits=logits), value
