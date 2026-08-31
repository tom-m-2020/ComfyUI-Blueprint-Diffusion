from __future__ import annotations

import math

import torch


class BlockDCTGeometry:
    HIGH_HW = (32, 64)
    GLOBAL_HW = (24, 48)
    BLOCK_HIGH = 4
    BLOCK_GLOBAL = 3
    TOLERANCE = 2e-6

    @staticmethod
    def _matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(size, device=device, dtype=torch.float64)
        frequencies = positions[:, None]
        matrix = torch.cos(
            math.pi / size * (positions[None, :] + 0.5) * frequencies
        )
        matrix[0] *= math.sqrt(1.0 / size)
        matrix[1:] *= math.sqrt(2.0 / size)
        return matrix.to(dtype=dtype)

    @staticmethod
    def _grid_to_blocks(value: torch.Tensor, size: int) -> torch.Tensor:
        batch, channels, height, width = value.shape
        if height % size or width % size:
            raise ValueError(
                f"Grid {tuple(value.shape[-2:])} is not divisible by {size}x{size}."
            )
        return value.reshape(
            batch, channels, height // size, size, width // size, size
        ).permute(0, 1, 2, 4, 3, 5)

    @staticmethod
    def _blocks_to_grid(blocks: torch.Tensor) -> torch.Tensor:
        batch, channels, blocks_y, blocks_x, local_h, local_w = blocks.shape
        return blocks.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, channels, blocks_y * local_h, blocks_x * local_w
        )

    def validate_high(self, value: torch.Tensor) -> None:
        if value.ndim != 4:
            raise ValueError(f"Blueprint requires a 4-D latent, got {value.ndim}-D.")
        if value.shape[0] != 1:
            raise ValueError(f"Blueprint requires batch size 1, got {value.shape[0]}.")
        if tuple(value.shape[-2:]) != self.HIGH_HW:
            raise ValueError(
                f"Blueprint requires latent grid {self.HIGH_HW}, got "
                f"{tuple(value.shape[-2:])}."
            )

    def restrict(self, value: torch.Tensor) -> torch.Tensor:
        self.validate_high(value)
        original_dtype = value.dtype
        work = value.float()
        blocks = self._grid_to_blocks(work, self.BLOCK_HIGH)
        q4 = self._matrix(4, work.device, work.dtype)
        q3 = self._matrix(3, work.device, work.dtype)
        coefficients = torch.matmul(torch.matmul(q4, blocks), q4.T)
        retained = coefficients[..., :3, :3]
        global_blocks = 0.75 * torch.matmul(torch.matmul(q3.T, retained), q3)
        return self._blocks_to_grid(global_blocks).to(dtype=original_dtype)

    def prolong(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[-2:]) != self.GLOBAL_HW:
            raise ValueError(
                f"Blueprint global grid must be [B,C,{self.GLOBAL_HW[0]},"
                f"{self.GLOBAL_HW[1]}], got {tuple(value.shape)}."
            )
        original_dtype = value.dtype
        work = value.float()
        blocks = self._grid_to_blocks(work, self.BLOCK_GLOBAL)
        q3 = self._matrix(3, work.device, work.dtype)
        q4 = self._matrix(4, work.device, work.dtype)
        retained = torch.matmul(torch.matmul(q3, blocks), q3.T)
        coefficients = torch.zeros(
            (*retained.shape[:-2], 4, 4), device=work.device, dtype=work.dtype
        )
        coefficients[..., :3, :3] = retained
        high_blocks = (4.0 / 3.0) * torch.matmul(
            torch.matmul(q4.T, coefficients), q4
        )
        return self._blocks_to_grid(high_blocks).to(dtype=original_dtype)

    def max_right_inverse_error(self, value: torch.Tensor) -> float:
        reconstructed = self.restrict(self.prolong(value))
        return float((reconstructed.float() - value.float()).abs().max())

    def qualify(self, *, device: torch.device, dtype: torch.dtype) -> float:
        generator = torch.Generator(device="cpu").manual_seed(314159)
        value = torch.randn((1, 3, *self.GLOBAL_HW), generator=generator).to(
            device=device, dtype=dtype
        )
        error = self.max_right_inverse_error(value)
        if error > self.TOLERANCE:
            raise RuntimeError(
                f"Block-DCT D(U(G)) qualification failed: {error} > "
                f"{self.TOLERANCE}."
            )
        return error
