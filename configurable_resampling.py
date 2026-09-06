from __future__ import annotations

from dataclasses import dataclass
import math
import time
import uuid
from typing import Any

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.samplers

from .adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from .regions import Region
from .terminal_resampling import (
    BlueprintRunState,
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    TerminalResamplingProcedure,
    initialize_blueprint,
    tensor_hash,
    validate_terminal_schedule,
)


@dataclass(frozen=True)
class ConfigurableResamplingGeometry:
    blueprint_hw: tuple[int, int]
    destination_hw: tuple[int, int]
    footprint_hw: tuple[int, int]
    stride_hw: tuple[int, int]
    working_hw: tuple[int, int]

    def validate(self) -> None:
        values = (*self.blueprint_hw, *self.destination_hw, *self.footprint_hw,
                  *self.stride_hw, *self.working_hw)
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Configurable Blueprint geometry requires positive integer dimensions.")
        if any(value < 16 or value > 64 for value in self.blueprint_hw) or math.prod(self.blueprint_hw) > 4096:
            raise ValueError("Blueprint G axes must be 16..64 with at most 4096 latent cells.")
        if any(value < 16 or value > 512 for value in self.destination_hw):
            raise ValueError("Destination H axes must be 16..512 latent cells.")
        if any(f > h for f, h in zip(self.footprint_hw, self.destination_hw)):
            raise ValueError("Destination footprint F must fit inside H.")
        if any(s > f for s, f in zip(self.stride_hw, self.footprint_hw)):
            raise ValueError("Stride must not exceed its footprint axis.")
        if any(value < 32 or value > 64 for value in self.working_hw) or math.prod(self.working_hw) > 4096:
            raise ValueError("Working W axes must be 32..64 with at most 4096 latent cells.")
        if any(w < f or w % f for w, f in zip(self.working_hw, self.footprint_hw)):
            raise ValueError("W must be an integer nearest-neighbor upscale of F on each axis.")

    @staticmethod
    def _starts(length: int, size: int, stride: int) -> tuple[int, ...]:
        final = length - size
        starts = list(range(0, final + 1, stride))
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def regions(self) -> tuple[Region, ...]:
        self.validate()
        ys = self._starts(self.destination_hw[0], self.footprint_hw[0], self.stride_hw[0])
        xs = self._starts(self.destination_hw[1], self.footprint_hw[1], self.stride_hw[1])
        return tuple(
            Region(index, y, x, *self.footprint_hw)
            for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
        )

    def is_frozen_oracle(self, refinement_sigma: float) -> bool:
        return (
            self == ConfigurableResamplingGeometry(
                blueprint_hw=(32, 64), destination_hw=(128, 256),
                footprint_hw=(32, 32), stride_hw=(24, 24), working_hw=(64, 64),
            )
            and refinement_sigma == 0.25
        )


def validate_refinement_sigma(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.10 <= value <= 0.50:
        raise ValueError("Configurable Blueprint refinement sigma must be within [0.10, 0.50].")
    return value


def initialize_configurable_blueprint(seed: int, geometry: ConfigurableResamplingGeometry,
                                     *, device, dtype=torch.float32) -> torch.Tensor:
    geometry.validate()
    qualified_blueprint = TerminalResamplingGeometry.QUALIFIED_PROFILES.get(geometry.destination_hw)
    if qualified_blueprint == geometry.blueprint_hw:
        return initialize_blueprint(
            seed,
            geometry=TerminalResamplingGeometry.for_destination(geometry.destination_hw),
            device=device,
            dtype=dtype,
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    construction = torch.randn((1, 128, *geometry.destination_hw), generator=generator, dtype=torch.float32)
    coarse = F.adaptive_avg_pool2d(construction, geometry.blueprint_hw)
    counts = []
    for destination, blueprint in zip(geometry.destination_hw, geometry.blueprint_hw):
        counts.append(torch.tensor([
            math.ceil((index + 1) * destination / blueprint)
            - math.floor(index * destination / blueprint)
            for index in range(blueprint)
        ], dtype=torch.float32))
    coarse_variance = counts[0].reciprocal()[:, None] * counts[1].reciprocal()[None, :]
    independent_generator = torch.Generator(device="cpu").manual_seed(int(seed) + 20_000_003)
    independent = torch.randn((1, 128, *geometry.blueprint_hw),
                              generator=independent_generator, dtype=torch.float32)
    result = coarse + torch.sqrt(1.0 - coarse_variance)[None, None] * independent
    return result.to(device=device, dtype=dtype)


def configurable_region_noise(seed: int, region: Region, working_hw: tuple[int, int],
                              *, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 22_000_003 + 1009 * region.index)
    value = torch.randn((1, 128, *working_hw), generator=generator, dtype=torch.float32)
    return value.to(device=device, dtype=dtype)


def _axis_coordinates(start: int, count: int, source: int, destination: int,
                      *, device) -> torch.Tensor:
    output = torch.arange(start, start + count, dtype=torch.float64, device=device)
    return ((output + 0.5) * source / destination - 0.5).clamp(0, source - 1)


def bounded_blueprint_transfer(value: torch.Tensor, geometry: ConfigurableResamplingGeometry,
                               region: Region) -> tuple[torch.Tensor, tuple[int, int]]:
    expected = (1, 128, *geometry.blueprint_hw)
    if tuple(value.shape) != expected:
        raise ValueError(f"Bounded Blueprint transfer expected {expected}, got {tuple(value.shape)}.")
    gy = _axis_coordinates(region.y, region.height, geometry.blueprint_hw[0],
                           geometry.destination_hw[0], device=value.device)
    gx = _axis_coordinates(region.x, region.width, geometry.blueprint_hw[1],
                           geometry.destination_hw[1], device=value.device)
    y0, y1 = int(torch.floor(gy.min())), int(torch.ceil(gy.max()))
    x0, x1 = int(torch.floor(gx.min())), int(torch.ceil(gx.max()))
    source = value[:, :, y0:y1 + 1, x0:x1 + 1]
    if source.shape[-2] < 2 or source.shape[-1] < 2:
        raise ValueError("Bounded bilinear transfer requires at least 2x2 G support.")
    ny = 2.0 * (gy - y0) / (source.shape[-2] - 1) - 1.0
    nx = 2.0 * (gx - x0) / (source.shape[-1] - 1) - 1.0
    grid_y, grid_x = torch.meshgrid(ny, nx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].to(dtype=value.dtype)
    footprint = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    scale = tuple(w // f for w, f in zip(geometry.working_hw, geometry.footprint_hw))
    working = F.interpolate(footprint, scale_factor=scale, mode="nearest")
    return working, (source.shape[-2], source.shape[-1])


def restrict_configurable_working(value: torch.Tensor,
                                  geometry: ConfigurableResamplingGeometry) -> torch.Tensor:
    expected = (1, 128, *geometry.working_hw)
    if tuple(value.shape) != expected:
        raise ValueError(f"Working restriction expected {expected}, got {tuple(value.shape)}.")
    scale = tuple(w // f for w, f in zip(geometry.working_hw, geometry.footprint_hw))
    return F.avg_pool2d(value, kernel_size=scale, stride=scale)


class StreamingOverlapMetric:
    def __init__(self) -> None:
        self.active: list[tuple[Region, torch.Tensor]] = []
        self.square_sum = 0.0
        self.count = 0

    def add(self, value: torch.Tensor, region: Region) -> None:
        current = value.detach().float().cpu()
        self.active = [(prior, prediction) for prior, prediction in self.active
                       if prior.y2 > region.y]
        for prior, prediction in self.active:
            y0, y1 = max(prior.y, region.y), min(prior.y2, region.y2)
            x0, x1 = max(prior.x, region.x), min(prior.x2, region.x2)
            if y0 >= y1 or x0 >= x1:
                continue
            left = prediction[..., y0-prior.y:y1-prior.y, x0-prior.x:x1-prior.x]
            right = current[..., y0-region.y:y1-region.y, x0-region.x:x1-region.x]
            delta = left - right
            self.square_sum += float(delta.square().sum())
            self.count += delta.numel()
        self.active.append((region, current))

    def finish(self) -> float:
        return 0.0 if self.count == 0 else math.sqrt(self.square_sum / self.count)


class ConfigurableResamplingProcedure(comfy.samplers.Sampler):
    HANDOFF_POLICY = "terminal_denoised"

    def __init__(self, *, seed: int, geometry: ConfigurableResamplingGeometry,
                 refinement_sigma: float, adapter: Any | None = None,
                 capture=None) -> None:
        self.seed = int(seed)
        self.geometry = geometry
        self.refinement_sigma = validate_refinement_sigma(refinement_sigma)
        self.adapter = adapter or Flux2TerminalResamplingAdapter()
        self.capture = capture
        self.telemetry: dict[str, Any] = {}
        self.run_id = uuid.uuid4().hex

    @staticmethod
    def _interrupt() -> None:
        comfy.model_management.throw_exception_if_processing_interrupted()

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        self.telemetry = {}
        self.geometry.validate()
        validate_terminal_schedule(sigmas)
        if self.geometry.is_frozen_oracle(self.refinement_sigma):
            oracle = TerminalResamplingProcedure(seed=self.seed, adapter=self.adapter,
                                                 geometry=TerminalResamplingGeometry())
            result = oracle.sample(model, sigmas, extra_args, callback, noise, latent_image,
                                   denoise_mask, disable_pbar)
            self.telemetry = {**oracle.telemetry, "execution": "configurable_exact_oracle_dispatch",
                              "handoff_policy": self.HANDOFF_POLICY,
                              "exact_terminal_oracle": True}
            return result
        if denoise_mask is not None:
            raise ValueError("Blueprint Configurable Prototype does not support masks.")
        expected = (1, 128, *self.geometry.destination_hw)
        if latent_image is None or tuple(latent_image.shape) != expected or tuple(noise.shape) != expected:
            raise ValueError(f"Blueprint Configurable Prototype requires destination {expected}.")
        if getattr(latent_image, "is_nested", False) or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Blueprint Configurable Prototype requires empty batch-one T2I latent input.")
        model_sampling = model.inner_model.model_sampling
        self.adapter.validate_prepared(
            guider=model, model_options=extra_args["model_options"], destination=latent_image,
            model_sampling=model_sampling, destination_hw=self.geometry.destination_hw,
        )
        device = latent_image.device
        state = BlueprintRunState(
            initialize_configurable_blueprint(self.seed, self.geometry, device=device),
            float(sigmas[0]), 0, f"{self.run_id}:initial",
        )
        if self.capture is not None:
            self.capture("initial_G", -1, state.blueprint)
        blueprint_calls = local_calls = 0
        blueprint_cuda_ms = local_cuda_ms = 0.0
        terminal_x0 = None
        assembler = None
        region_records = []
        overlap_metric = StreamingOverlapMetric()
        barriers = []
        peaks = []
        started = time.perf_counter()
        try:
            for ordinal in range(4):
                self._interrupt()
                sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
                snapshot = state.blueprint.clone()
                begin = end = None
                if state.blueprint.is_cuda:
                    begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
                x0 = self.adapter.predict_native(
                    guider=model, value=state.blueprint, sigma=sigma,
                    expected_hw=self.geometry.blueprint_hw,
                    model_options=extra_args["model_options"], seed=self.seed,
                )
                if end is not None:
                    end.record(); torch.cuda.synchronize(device); blueprint_cuda_ms += float(begin.elapsed_time(end))
                blueprint_calls += 1
                if not torch.equal(state.blueprint, snapshot):
                    raise RuntimeError("Configurable Blueprint model prediction mutated accepted G.")
                if tuple(x0.shape) != tuple(state.blueprint.shape) or not bool(x0.isfinite().all()):
                    raise RuntimeError(f"Invalid configurable Blueprint prediction at interval {ordinal}.")
                proposal = state.blueprint + (sigma_next-sigma) * (state.blueprint-x0) / sigma
                terminal_x0 = x0
                if self.capture is not None:
                    self.capture("x0_G", ordinal, x0)
                state = BlueprintRunState(proposal, float(sigma_next), ordinal + 1,
                                          f"{self.run_id}:{ordinal}:accepted")
                if self.capture is not None:
                    self.capture("accepted_G", ordinal, state.blueprint)
                if callback is not None:
                    callback(ordinal, x0, state.blueprint, 5)

            self._interrupt()
            terminal_cpu = terminal_x0.detach().float().cpu()
            regions = self.geometry.regions()
            assembler = StreamingOverlapAssembler(regions=regions,
                                                   target_hw=self.geometry.destination_hw,
                                                   template=latent_image)
            local_sigma = torch.tensor(self.refinement_sigma, device=device, dtype=terminal_x0.dtype)
            if latent_image.is_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            for region in regions:
                self._interrupt()
                anchor, source_hw = bounded_blueprint_transfer(terminal_cpu, self.geometry, region)
                anchor = anchor.to(device=device, dtype=terminal_x0.dtype)
                epsilon = configurable_region_noise(self.seed, region, self.geometry.working_hw,
                                                    device=device, dtype=anchor.dtype)
                working = model_sampling.noise_scaling(local_sigma, epsilon, anchor, False)
                snapshot = working.clone()
                begin = end = None
                if working.is_cuda:
                    begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
                x0_w = self.adapter.predict_native(
                    guider=model, value=working, sigma=local_sigma,
                    expected_hw=self.geometry.working_hw,
                    model_options=extra_args["model_options"], seed=self.seed,
                )
                if end is not None:
                    end.record(); torch.cuda.synchronize(device); local_cuda_ms += float(begin.elapsed_time(end))
                local_calls += 1
                if not torch.equal(working, snapshot):
                    raise RuntimeError(f"Local model call mutated W for region {region.index}.")
                if tuple(x0_w.shape) != tuple(working.shape) or not bool(x0_w.isfinite().all()):
                    raise RuntimeError(f"Invalid configurable local prediction for region {region.index}.")
                restricted = restrict_configurable_working(x0_w, self.geometry)
                assembler.add(restricted, region)
                overlap_metric.add(restricted, region)
                region_records.append({
                    "index": region.index, "rect": (region.y, region.x, region.height, region.width),
                    "seed": self.seed + 22_000_003 + 1009 * region.index,
                    "bounded_source_hw": source_hw, "noise_hash": tensor_hash(epsilon),
                    "working_hash": tensor_hash(working), "x0_W_hash": tensor_hash(x0_w),
                    "restricted_hash": tensor_hash(restricted),
                })
                if working.is_cuda:
                    peaks.append(int(torch.cuda.max_memory_allocated(device)))
                del anchor, epsilon, working, snapshot, x0_w, restricted
                if latent_image.is_cuda:
                    torch.cuda.synchronize(device); barriers.append(int(torch.cuda.memory_allocated(device)))
                self._interrupt()
            final_h, coverage = assembler.finish()
            if callback is not None:
                callback(4, final_h, final_h, 5)
            self.telemetry = {
                "execution": "configurable_terminal_resampling", "handoff_policy": self.HANDOFF_POLICY,
                "exact_terminal_oracle": False, "blueprint_predictions": blueprint_calls,
                "local_predictions": local_calls, "destination_model_predictions": 0,
                "blueprint_hw": self.geometry.blueprint_hw, "destination_hw": self.geometry.destination_hw,
                "footprint_hw": self.geometry.footprint_hw, "stride_hw": self.geometry.stride_hw,
                "overlap_hw": tuple(f-s for f, s in zip(self.geometry.footprint_hw, self.geometry.stride_hw)),
                "working_hw": self.geometry.working_hw, "refinement_sigmas": (self.refinement_sigma, 0.0),
                "blueprint_interpolation": "bounded_bilinear", "footprint_to_working": "nearest",
                "working_to_footprint": "area_mean", "regions": region_records,
                "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max()),
                "overlap_rms": overlap_metric.finish(),
                "final_H_hash": tensor_hash(final_h), "blueprint_cuda_ms": blueprint_cuda_ms,
                "local_cuda_ms": local_cuda_ms, "wall_seconds": time.perf_counter()-started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if final_h.is_cuda else 0,
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if final_h.is_cuda else 0,
                "region_peak_allocated_bytes": tuple(peaks),
                "region_barrier_allocated_bytes": tuple(barriers),
                "local_step_count": 1, "interpolation_policy": "bilinear_nearest_mean",
            }
            return final_h
        finally:
            terminal_x0 = None
            assembler = None
