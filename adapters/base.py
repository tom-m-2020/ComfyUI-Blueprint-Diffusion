from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..regions import Region


@dataclass(frozen=True)
class WorkEstimate:
    global_tokens: int
    local_tokens: int
    model_predictions: int


class ModelFamilyAdapter(Protocol):
    family: str

    def validate_run(
        self,
        *,
        guider: Any,
        high_shape: tuple[int, ...],
        global_shape: tuple[int, ...],
        crops: tuple[Region, ...],
        sigmas: torch.Tensor,
        latent: torch.Tensor,
    ) -> None: ...

    def predict_global(
        self,
        *,
        guider: Any,
        g: torch.Tensor,
        sigma: torch.Tensor,
        canvas: tuple[int, int],
        model_options: dict[str, Any],
        seed: int,
    ) -> torch.Tensor: ...

    def predict_region(
        self,
        *,
        guider: Any,
        h_view: torch.Tensor,
        sigma: torch.Tensor,
        canvas: tuple[int, int],
        region: Region,
        model_options: dict[str, Any],
        seed: int,
    ) -> torch.Tensor: ...

    def describe_work(
        self, *, global_shape: tuple[int, ...], crops: tuple[Region, ...]
    ) -> WorkEstimate: ...
