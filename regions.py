from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Region:
    index: int
    y: int
    x: int
    height: int = 32
    width: int = 32

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def x2(self) -> int:
        return self.x + self.width


class FixedCropPlanner:
    CROP_SIZE = 32
    STRIDE = 24
    QUALIFIED_GEOMETRIES = {
        (64, 128): (64, 48),
        (48, 96): (48, 36),
    }

    @staticmethod
    def _starts(length: int, crop_size: int = 32, stride: int = 24) -> tuple[int, ...]:
        if length < crop_size:
            raise ValueError(
                f"Blueprint target latent axis {length} cannot fit the qualified "
                f"{crop_size}x{crop_size} local crop."
            )
        final = length - crop_size
        starts = list(range(0, final + 1, stride))
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def plan(self, target_hw: tuple[int, int]) -> tuple[Region, ...]:
        target_hw = tuple(int(value) for value in target_hw)
        if len(target_hw) != 2 or any(value % 4 for value in target_hw):
            raise ValueError(
                "Blueprint target latent axes must both be divisible by 4, got "
                f"{target_hw}."
            )
        crop_size, stride = self.QUALIFIED_GEOMETRIES.get(
            target_hw, (self.CROP_SIZE, self.STRIDE)
        )
        ys = self._starts(target_hw[0], crop_size, stride)
        xs = self._starts(target_hw[1], crop_size, stride)
        return tuple(
            Region(index, y, x, crop_size, crop_size)
            for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
        )


class OverlapAssembler:
    @staticmethod
    def _axis_weight(
        start: int,
        end: int,
        starts: list[int],
        ends: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        weight = torch.ones(end - start, dtype=torch.float32, device=device)
        previous_ends = [value for value in ends if start < value < end]
        next_starts = [value for value in starts if start < value < end]
        if previous_ends:
            ramp = min(previous_ends) - start
            if ramp > 0:
                weight[:ramp] = torch.linspace(
                    0.0, 1.0, ramp + 2, device=device
                )[1:-1]
        if next_starts:
            ramp = end - max(next_starts)
            if ramp > 0:
                weight[-ramp:] = torch.linspace(
                    1.0, 0.0, ramp + 2, device=device
                )[1:-1]
        return weight

    def weight(
        self, region: Region, regions: tuple[Region, ...], device: torch.device
    ) -> torch.Tensor:
        row = [item for item in regions if item.y == region.y]
        column = [item for item in regions if item.x == region.x]
        wy = self._axis_weight(
            region.y,
            region.y2,
            [item.y for item in column],
            [item.y2 for item in column],
            device,
        )
        wx = self._axis_weight(
            region.x,
            region.x2,
            [item.x for item in row],
            [item.x2 for item in row],
            device,
        )
        return wy[:, None] * wx[None, :]

    def assemble(
        self,
        predictions: list[torch.Tensor],
        regions: tuple[Region, ...],
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not predictions or len(predictions) != len(regions):
            raise ValueError(
                "Blueprint requires one prediction for every planned crop."
            )
        first = predictions[0]
        output = torch.zeros(
            (first.shape[0], first.shape[1], *target_hw),
            dtype=first.dtype,
            device=first.device,
        )
        coverage = torch.zeros(
            (first.shape[0], 1, *target_hw),
            dtype=torch.float32,
            device=first.device,
        )
        for prediction, region in zip(predictions, regions):
            if region.y < 0 or region.x < 0 or region.y2 > target_hw[0] or region.x2 > target_hw[1]:
                raise ValueError(
                    f"Crop {region.index} lies outside target grid {target_hw}."
                )
            expected = (region.height, region.width)
            if tuple(prediction.shape[-2:]) != expected:
                raise ValueError(
                    f"Crop {region.index} prediction has "
                    f"{tuple(prediction.shape[-2:])}, expected {expected}."
                )
            weight = self.weight(region, regions, output.device)[None, None]
            output[:, :, region.y:region.y2, region.x:region.x2] += (
                prediction * weight
            )
            coverage[:, :, region.y:region.y2, region.x:region.x2] += weight
        if not bool(torch.isfinite(coverage).all()) or float(coverage.min()) <= 0.0:
            raise RuntimeError("Blueprint crop assembly has incomplete coverage.")
        return output / coverage, coverage
