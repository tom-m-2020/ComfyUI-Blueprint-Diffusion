from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry.block_dct import BlockDCTGeometry


@dataclass(frozen=True)
class Acceptance:
    g: torch.Tensor
    h: torch.Tensor
    terminal_release: bool
    projection_rms: float | None
    global_synchronized: bool


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
        if sigma_next == 0.0:
            raise ValueError("Terminal release must use accept_terminal without a global proposal.")
        delta = g_star - geometry.restrict(h_star)
        projection = geometry.prolong(delta)
        h_next = h_star + projection
        return Acceptance(
            g_star,
            h_next,
            False,
            float(projection.float().square().mean().sqrt()),
            True,
        )

    def accept_terminal(
        self,
        *,
        retained_g: torch.Tensor,
        h_star: torch.Tensor,
        sigma_next: float,
    ) -> Acceptance:
        if sigma_next != 0.0:
            raise ValueError("Terminal acceptance requires sigma_next exactly zero.")
        return Acceptance(retained_g, h_star, True, None, False)


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
