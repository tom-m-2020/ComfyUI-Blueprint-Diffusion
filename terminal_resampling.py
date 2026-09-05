from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
import uuid
from typing import Any, Callable

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers

from .adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from .regions import OverlapAssembler, Region


QUALIFIED_SIGMAS = (
    1.0,
    0.9991771578788757,
    0.9975355267524719,
    0.9926428198814392,
    0.0,
)


@dataclass(frozen=True)
class BlueprintRunState:
    blueprint: torch.Tensor
    sigma: float
    ordinal: int
    accepted_evaluation_id: str


@dataclass(frozen=True)
class TerminalResamplingGeometry:
    blueprint_hw: tuple[int, int] = (32, 64)
    destination_hw: tuple[int, int] = (128, 256)
    region_hw: tuple[int, int] = (32, 32)
    stride_hw: tuple[int, int] = (24, 24)
    working_hw: tuple[int, int] = (64, 64)

    QUALIFIED_PROFILES = {
        (128, 256): (32, 64),
        (128, 192): (36, 54),
        (128, 128): (45, 45),
        (256, 128): (64, 32),
    }

    @classmethod
    def for_destination(cls, destination_hw: tuple[int, int]) -> "TerminalResamplingGeometry":
        destination_hw = tuple(int(value) for value in destination_hw)
        try:
            blueprint_hw = cls.QUALIFIED_PROFILES[destination_hw]
        except KeyError as error:
            supported = ", ".join(f"{h}x{w}" for h, w in cls.QUALIFIED_PROFILES)
            raise ValueError(
                f"Unsupported terminal-resampling destination {destination_hw}; "
                f"qualified latent geometries are {supported}."
            ) from error
        return cls(blueprint_hw=blueprint_hw, destination_hw=destination_hw)

    def validate(self) -> None:
        expected_blueprint = self.QUALIFIED_PROFILES.get(self.destination_hw)
        if (
            expected_blueprint != self.blueprint_hw
            or self.region_hw != (32, 32)
            or self.stride_hw != (24, 24)
            or self.working_hw != (64, 64)
        ):
            raise ValueError(
                "Blueprint Terminal Resampling requires a qualified finite geometry "
                "profile with region=32x32/stride24 and W=64x64."
            )

    @staticmethod
    def _starts(length: int, size: int, stride: int) -> tuple[int, ...]:
        final = length - size
        starts = list(range(0, final + 1, stride))
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def regions(self) -> tuple[Region, ...]:
        self.validate()
        ys = self._starts(self.destination_hw[0], self.region_hw[0], self.stride_hw[0])
        xs = self._starts(self.destination_hw[1], self.region_hw[1], self.stride_hw[1])
        regions = tuple(
            Region(index, y, x, *self.region_hw)
            for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
        )
        return regions


def validate_terminal_schedule(sigmas: torch.Tensor) -> None:
    if sigmas.ndim != 1 or sigmas.numel() != len(QUALIFIED_SIGMAS):
        raise ValueError("Blueprint Terminal Resampling requires exactly five sigma values.")
    values = tuple(float(value) for value in sigmas.detach().float().cpu())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Blueprint Terminal Resampling sigmas must be finite.")
    if values != QUALIFIED_SIGMAS:
        raise ValueError(
            "Blueprint Terminal Resampling requires the exact qualified Phase-25 sigma schedule."
        )


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def initialize_blueprint(
    seed: int,
    *,
    geometry: TerminalResamplingGeometry | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    geometry = geometry or TerminalResamplingGeometry()
    geometry.validate()
    high_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    construction = torch.randn(
        (1, 128, *geometry.destination_hw), generator=high_generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    divisible = all(
        destination % blueprint == 0
        for destination, blueprint in zip(geometry.destination_hw, geometry.blueprint_hw)
    )
    if divisible:
        scale_y = geometry.destination_hw[0] // geometry.blueprint_hw[0]
        scale_x = geometry.destination_hw[1] // geometry.blueprint_hw[1]
        coarse = F.avg_pool2d(construction, (scale_y, scale_x), (scale_y, scale_x))
        coarse_variance = 1.0 / (scale_y * scale_x)
    else:
        coarse = F.adaptive_avg_pool2d(construction, geometry.blueprint_hw)
        counts = []
        for destination, blueprint in zip(geometry.destination_hw, geometry.blueprint_hw):
            axis_counts = [
                math.ceil((index + 1) * destination / blueprint)
                - math.floor(index * destination / blueprint)
                for index in range(blueprint)
            ]
            counts.append(torch.tensor(axis_counts, device=device, dtype=dtype))
        coarse_variance = (
            counts[0].reciprocal()[:, None] * counts[1].reciprocal()[None, :]
        )[None, None]
    del construction
    blueprint_generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 20_000_003
    )
    independent = torch.randn(
        (1, 128, *geometry.blueprint_hw), generator=blueprint_generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    if isinstance(coarse_variance, float):
        independent_scale = math.sqrt(1.0 - coarse_variance)
    else:
        independent_scale = torch.sqrt(1.0 - coarse_variance)
    blueprint = coarse + independent_scale * independent
    return blueprint


def region_seed(seed: int, region: Region) -> int:
    return int(seed) + 22_000_003 + 1009 * int(region.index)


def region_noise(
    seed: int,
    region: Region,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(region_seed(seed, region))
    value = torch.randn((1, 128, 64, 64), generator=generator, dtype=torch.float32)
    return value.to(device=device, dtype=dtype)


def map_blueprint_to_destination(
    value: torch.Tensor, geometry: TerminalResamplingGeometry
) -> torch.Tensor:
    if tuple(value.shape) != (1, 128, *geometry.blueprint_hw):
        raise ValueError(f"Unexpected Blueprint prediction shape {tuple(value.shape)}.")
    return F.interpolate(
        value, size=geometry.destination_hw, mode="bilinear", align_corners=False
    )


def lift_region(anchor: torch.Tensor) -> torch.Tensor:
    if tuple(anchor.shape) != (1, 128, 32, 32):
        raise ValueError(f"Unexpected destination anchor shape {tuple(anchor.shape)}.")
    return F.interpolate(anchor, scale_factor=2.0, mode="nearest")


def build_working_canvas(
    *, model_sampling, anchor: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, float]:
    if tuple(anchor.shape) != (1, 128, 64, 64) or noise.shape != anchor.shape:
        raise ValueError("Blueprint terminal working tensors must be [1,128,64,64].")
    sigma = torch.tensor(0.25, device=anchor.device, dtype=anchor.dtype)
    working = model_sampling.noise_scaling(sigma, noise, anchor, False)
    expected = 0.75 * anchor + 0.25 * noise
    error = float((working.float() - expected.float()).abs().max())
    if error != 0.0:
        raise RuntimeError(
            f"CONST sigma-0.25 construction diverged from the qualified rule: {error}."
        )
    return working, error


def restrict_working_prediction(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != (1, 128, 64, 64):
        raise ValueError(f"Unexpected working prediction shape {tuple(value.shape)}.")
    return F.avg_pool2d(value, 2, 2)


class StreamingOverlapAssembler:
    def __init__(
        self,
        *,
        regions: tuple[Region, ...],
        target_hw: tuple[int, int],
        template: torch.Tensor,
    ) -> None:
        self.regions = regions
        self.target_hw = target_hw
        self.next_index = 0
        self.weights = OverlapAssembler()
        self.weighted_sum = torch.zeros(
            (template.shape[0], template.shape[1], *target_hw),
            dtype=template.dtype,
            device=template.device,
        )
        self.coverage = torch.zeros(
            (template.shape[0], 1, *target_hw),
            dtype=torch.float32,
            device=template.device,
        )

    def add(self, prediction: torch.Tensor, region: Region) -> None:
        if region.index != self.next_index or region != self.regions[self.next_index]:
            raise RuntimeError(
                f"Terminal region order drifted: expected {self.next_index}, got {region.index}."
            )
        if tuple(prediction.shape) != (1, 128, region.height, region.width):
            raise ValueError(f"Region {region.index} has invalid prediction shape.")
        weight = self.weights.weight(region, self.regions, prediction.device)[None, None]
        self.weighted_sum[:, :, region.y:region.y2, region.x:region.x2] += prediction * weight
        self.coverage[:, :, region.y:region.y2, region.x:region.x2] += weight
        self.next_index += 1

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.next_index != len(self.regions):
            raise RuntimeError(
                f"Terminal assembly received {self.next_index}/{len(self.regions)} regions."
            )
        if not bool(torch.isfinite(self.coverage).all()) or float(self.coverage.min()) <= 0.0:
            raise RuntimeError("Terminal resampling has incomplete crop coverage.")
        output = self.weighted_sum / self.coverage
        if tuple(output.shape) != (1, 128, *self.target_hw) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Terminal resampling produced an invalid final destination.")
        return output, self.coverage


class TerminalResamplingProcedure(comfy.samplers.Sampler):
    def __init__(
        self,
        *,
        seed: int,
        capture: Callable[[str, int, torch.Tensor], None] | None = None,
        adapter: Any | None = None,
        geometry: TerminalResamplingGeometry | None = None,
    ) -> None:
        self.seed = int(seed)
        self.capture = capture
        self.adapter = adapter or Flux2TerminalResamplingAdapter()
        self.geometry = geometry or TerminalResamplingGeometry()
        self.telemetry: dict[str, Any] = {}
        self.run_id = uuid.uuid4().hex

    @staticmethod
    def _interrupt() -> None:
        comfy.model_management.throw_exception_if_processing_interrupted()

    def _capture(self, name: str, ordinal: int, value: torch.Tensor) -> None:
        if self.capture is not None:
            self.capture(name, ordinal, value)

    def sample(
        self,
        model,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        del disable_pbar
        self.telemetry = {}
        self.geometry.validate()
        validate_terminal_schedule(sigmas)
        if denoise_mask is not None:
            raise ValueError("Blueprint Terminal Resampling does not support masks.")
        expected_destination_shape = (1, 128, *self.geometry.destination_hw)
        if latent_image is None or tuple(latent_image.shape) != expected_destination_shape:
            raise ValueError(
                f"Blueprint Terminal Resampling requires destination {expected_destination_shape}."
            )
        if getattr(latent_image, "is_nested", False) or getattr(noise, "is_nested", False):
            raise ValueError("Blueprint Terminal Resampling does not support nested latents.")
        if bool(torch.count_nonzero(latent_image)):
            raise ValueError("Blueprint Terminal Resampling supports only empty-latent T2I.")
        if tuple(noise.shape) != tuple(latent_image.shape):
            raise ValueError("Blueprint Terminal Resampling received incompatible procedure noise.")
        model_sampling = model.inner_model.model_sampling
        self.adapter.validate_prepared(
            guider=model,
            model_options=extra_args["model_options"],
            destination=latent_image,
            model_sampling=model_sampling,
            destination_hw=self.geometry.destination_hw,
        )

        device = latent_image.device
        blueprint = initialize_blueprint(
            self.seed, geometry=self.geometry, device=device, dtype=torch.float32
        )
        state = BlueprintRunState(blueprint, float(sigmas[0]), 0, f"{self.run_id}:initial")
        self._capture("initial_B", -1, state.blueprint)
        blueprint_calls = 0
        local_calls = 0
        blueprint_cuda_ms = 0.0
        local_cuda_ms = 0.0
        terminal_x0 = None
        mapped = None
        assembler = None
        region_records: list[dict[str, Any]] = []
        peak_by_region: list[int] = []
        barrier_allocated: list[int] = []
        started = time.perf_counter()
        try:
            for ordinal in range(4):
                self._interrupt()
                sigma = sigmas[ordinal]
                sigma_next = sigmas[ordinal + 1]
                if state.ordinal != ordinal or state.sigma != float(sigma):
                    raise RuntimeError("Blueprint accepted-state ordinal/sigma drifted.")
                snapshot = state.blueprint.clone()
                begin = end = None
                if state.blueprint.is_cuda:
                    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                    begin.record()
                x0 = self.adapter.predict_native(
                    guider=model,
                    value=state.blueprint,
                    sigma=sigma,
                    expected_hw=self.geometry.blueprint_hw,
                    model_options=extra_args["model_options"],
                    seed=self.seed,
                )
                if end is not None:
                    end.record(); torch.cuda.synchronize(); blueprint_cuda_ms += float(begin.elapsed_time(end))
                blueprint_calls += 1
                if not torch.equal(state.blueprint, snapshot):
                    raise RuntimeError("Blueprint model prediction mutated accepted state.")
                if tuple(x0.shape) != tuple(state.blueprint.shape) or not bool(torch.isfinite(x0).all()):
                    raise RuntimeError(f"Invalid Blueprint prediction at interval {ordinal}.")
                proposal = state.blueprint + (sigma_next - sigma) * (state.blueprint - x0) / sigma
                if not bool(torch.isfinite(proposal).all()):
                    raise RuntimeError(f"Nonfinite Blueprint proposal at interval {ordinal}.")
                terminal_x0 = x0
                self._capture("x0_B", ordinal, x0)
                state = BlueprintRunState(
                    proposal,
                    float(sigma_next),
                    ordinal + 1,
                    f"{self.run_id}:{ordinal}:accepted",
                )
                self._capture("accepted_B", ordinal, state.blueprint)
                if callback is not None:
                    callback(ordinal, x0, state.blueprint, 5)

            self._interrupt()
            # Phase 25 detached the terminal estimate before the qualified
            # bilinear map. Preserve that CPU arithmetic exactly, then return
            # the mapped anchor to the model device for streamed local work.
            mapped = map_blueprint_to_destination(
                terminal_x0.detach().float().cpu(), self.geometry
            ).to(device=device, dtype=terminal_x0.dtype)
            self._capture("mapped_terminal_B", 3, mapped)
            regions = self.geometry.regions()
            assembler = StreamingOverlapAssembler(
                regions=regions,
                target_hw=self.geometry.destination_hw,
                template=mapped,
            )
            local_sigma = torch.tensor(0.25, device=device, dtype=mapped.dtype)
            if mapped.is_cuda:
                torch.cuda.reset_peak_memory_stats(mapped.device)
            for region in regions:
                self._interrupt()
                anchor_view = mapped[:, :, region.y:region.y2, region.x:region.x2]
                anchor = lift_region(anchor_view)
                epsilon = region_noise(
                    self.seed, region, device=device, dtype=anchor.dtype
                )
                working, construction_error = build_working_canvas(
                    model_sampling=model_sampling, anchor=anchor, noise=epsilon
                )
                working_hash = tensor_hash(working)
                noise_hash = tensor_hash(epsilon)
                snapshot = working.clone()
                begin = end = None
                if working.is_cuda:
                    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                    begin.record()
                x0_w = self.adapter.predict_native(
                    guider=model,
                    value=working,
                    sigma=local_sigma,
                    expected_hw=self.geometry.working_hw,
                    model_options=extra_args["model_options"],
                    seed=self.seed,
                )
                if end is not None:
                    end.record(); torch.cuda.synchronize(); local_cuda_ms += float(begin.elapsed_time(end))
                local_calls += 1
                if not torch.equal(working, snapshot):
                    raise RuntimeError(f"Local model call mutated W for region {region.index}.")
                if tuple(x0_w.shape) != tuple(working.shape) or not bool(torch.isfinite(x0_w).all()):
                    raise RuntimeError(f"Invalid local prediction for region {region.index}.")
                prediction = restrict_working_prediction(x0_w)
                assembler.add(prediction, region)
                region_records.append({
                    "index": region.index,
                    "rect": (region.y, region.x, region.height, region.width),
                    "seed": region_seed(self.seed, region),
                    "noise_hash": noise_hash,
                    "working_hash": working_hash,
                    "x0_W_hash": tensor_hash(x0_w),
                    "restricted_hash": tensor_hash(prediction),
                    "construction_max_abs": construction_error,
                })
                if working.is_cuda:
                    peak_by_region.append(int(torch.cuda.max_memory_allocated(working.device)))
                del anchor_view, anchor, epsilon, working, snapshot, x0_w, prediction
                if mapped.is_cuda:
                    torch.cuda.synchronize(mapped.device)
                    barrier_allocated.append(int(torch.cuda.memory_allocated(mapped.device)))
                self._interrupt()
            final_h, coverage = assembler.finish()
            self._capture("final_H", 4, final_h)
            if callback is not None:
                callback(4, final_h, final_h, 5)
            self.telemetry = {
                "execution": "terminal_resampling_first_slice",
                "blueprint_predictions": blueprint_calls,
                "local_predictions": local_calls,
                "destination_model_predictions": 0,
                "blueprint_hw": self.geometry.blueprint_hw,
                "destination_hw": self.geometry.destination_hw,
                "working_hw": self.geometry.working_hw,
                "blueprint_tokens": self.geometry.blueprint_hw[0] * self.geometry.blueprint_hw[1],
                "regions": region_records,
                "coverage_min": float(coverage.min()),
                "coverage_max": float(coverage.max()),
                "final_H_hash": tensor_hash(final_h),
                "blueprint_cuda_ms": blueprint_cuda_ms,
                "local_cuda_ms": local_cuda_ms,
                "wall_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if final_h.is_cuda else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if final_h.is_cuda else 0,
                "region_peak_allocated_bytes": tuple(peak_by_region),
                "region_barrier_allocated_bytes": tuple(barrier_allocated),
                "block_dct_calls": 0,
                "persistent_H_states": 0,
                "hard_terminal_policy_calls": 0,
                "flux2_block_executor_calls": 0,
                "global_kv_context_calls": 0,
                "periodic_resampling_calls": 0,
                "post_anchor_calls": 0,
            }
            return final_h
        finally:
            terminal_x0 = None
            mapped = None
            assembler = None
