from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlueprintState:
    g: torch.Tensor
    h: torch.Tensor
    sigma: float
    ordinal: int
    accepted_evaluation_id: str
