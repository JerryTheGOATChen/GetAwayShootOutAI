"""The one network both training methods use, so the comparison is about how they learn, not what they learn with."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sim.env import OBS_SIZE

N_ACTIONS = 3  # release, hold left, hold right
HIDDEN = 64


class Policy(nn.Module):
    """53 inputs -> 64 -> 64 -> a score for each of the 3 actions."""

    def __init__(self, obs_size: int = OBS_SIZE, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """The action a trained agent plays: the highest-scoring one, no randomness."""
        return int(self.net(torch.from_numpy(obs)).argmax())


def save_agent(path, policy: Policy, **info) -> None:
    """Agents from either method are the same kind of file: the policy's weights plus a few facts about it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"policy": policy.state_dict(), "info": info}, tmp)
    tmp.replace(path)  # never leaves a half-written file behind if training is stopped mid-save


def load_agent(path) -> tuple[Policy, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    policy = Policy()
    policy.load_state_dict(data["policy"])
    policy.eval()
    return policy, data.get("info", {})
