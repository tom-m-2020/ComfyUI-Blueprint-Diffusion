from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry.block_dct import BlockDCTGeometry


@dataclass(frozen=True)
class Acceptance:
    g: torch.Tensor
    h: torch.Tensor
    terminal_release: bool
    projection_rms: float


class HardNonterminalTerminalRelease:
    def accept(
        self,
        *,
        g_star: torch.Tensor,
        h_star: torch.Tensor,
        sigma_next: float,
        geometry: BlockDCTGeometry,
    ) -> Acceptance:
        if not math_isfinite(sigma_next) or sigma_next < 0.0:
            raise ValueError(f"Invalid sigma_next {sigma_next}.")
        delta = g_star - geometry.restrict(h_star)
        projection = geometry.prolong(delta)
        if sigma_next == 0.0:
            return Acceptance(g_star, h_star, True, float(projection.float().square().mean().sqrt()))
        h_next = h_star + projection
        return Acceptance(g_star, h_next, False, float(projection.float().square().mean().sqrt()))


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
