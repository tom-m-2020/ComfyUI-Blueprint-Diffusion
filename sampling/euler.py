from __future__ import annotations

import math
import uuid
from typing import Any, Callable

import torch

import comfy.samplers

from ..adapters.flux2 import Flux2Adapter
from ..geometry.block_dct import BlockDCTGeometry
from ..policies import HardNonterminalTerminalRelease
from ..regions import FixedCropPlanner, OverlapAssembler
from ..state import BlueprintState


TensorCapture = Callable[[str, int, torch.Tensor], None]


def _summary(value: torch.Tensor) -> dict[str, Any]:
    work = value.float()
    return {
        "shape": tuple(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


def validate_schedule(sigmas: torch.Tensor) -> None:
    if sigmas.ndim != 1 or sigmas.numel() != 5:
        raise ValueError("Blueprint first slice requires exactly four Euler intervals.")
    values = sigmas.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Blueprint sigma schedule must be finite.")
    if float(values[-1]) != 0.0:
        raise ValueError("Blueprint sigma schedule must terminate at exactly zero.")
    if float(values[0]) != 1.0:
        raise ValueError(
            "Blueprint first slice requires sigma[0] exactly 1.0; partial "
            "denoise schedules are unsupported."
        )
    if bool((values[:-1] <= 0.0).any()) or bool((values[1:] >= values[:-1]).any()):
        raise ValueError("Blueprint sigmas must be positive then strictly decreasing.")


class BlueprintCoordinator:
    def __init__(self, *, capture: TensorCapture | None = None) -> None:
        self.geometry = BlockDCTGeometry()
        self.planner = FixedCropPlanner()
        self.assembler = OverlapAssembler()
        self.policy = HardNonterminalTerminalRelease()
        self.adapter = Flux2Adapter()
        self.capture = capture
        self.telemetry: list[dict[str, Any]] = []
        self.run_id = uuid.uuid4().hex

    def _capture(self, name: str, ordinal: int, value: torch.Tensor) -> None:
        if self.capture is not None:
            self.capture(name, ordinal, value)

    def initialize(self, h: torch.Tensor, sigma: torch.Tensor) -> BlueprintState:
        self.geometry.validate_high(h)
        error = self.geometry.qualify(device=h.device, dtype=h.dtype)
        g = self.geometry.restrict(h)
        self._capture("initial_H", -1, h)
        self._capture("initial_G", -1, g)
        self.telemetry.append({
            "event": "initialize",
            "right_inverse_max_abs": error,
            "H": _summary(h),
            "G": _summary(g),
        })
        return BlueprintState(g, h, float(sigma), 0, f"{self.run_id}:initial")

    def evaluate(
        self,
        *,
        guider,
        state: BlueprintState,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        model_options: dict[str, Any],
        seed: int,
    ) -> tuple[BlueprintState, torch.Tensor]:
        ordinal = state.ordinal
        sigma_value = float(sigma)
        sigma_next_value = float(sigma_next)
        if sigma_value != state.sigma:
            raise RuntimeError(
                f"Accepted sigma {state.sigma} does not match interval sigma {sigma_value}."
            )
        regions = self.planner.plan(tuple(state.h.shape[-2:]))
        self.adapter.validate_run(
            guider=guider,
            high_shape=tuple(state.h.shape),
            global_shape=tuple(state.g.shape),
            crops=regions,
            sigmas=torch.stack((sigma, sigma_next)),
            latent=state.h,
        )

        accepted_h_snapshot = state.h.clone()
        accepted_g_snapshot = state.g.clone()
        x0_g = self.adapter.predict_global(
            guider=guider,
            g=state.g,
            sigma=sigma,
            canvas=self.geometry.HIGH_HW,
            model_options=model_options,
            seed=seed,
        )
        local_predictions = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            if view.untyped_storage().data_ptr() != state.h.untyped_storage().data_ptr():
                raise RuntimeError(f"Crop {region.index} is not a view of accepted H.")
            local_predictions.append(self.adapter.predict_region(
                guider=guider,
                h_view=view,
                sigma=sigma,
                canvas=self.geometry.HIGH_HW,
                region=region,
                model_options=model_options,
                seed=seed,
            ))
        if not torch.equal(state.h, accepted_h_snapshot) or not torch.equal(state.g, accepted_g_snapshot):
            raise RuntimeError("A Blueprint model call mutated accepted state.")

        x0_h, coverage = self.assembler.assemble(
            local_predictions, regions, self.geometry.HIGH_HW
        )
        dt = sigma_next - sigma
        g_star = state.g + (state.g - x0_g) / sigma * dt
        h_star = state.h + (state.h - x0_h) / sigma * dt
        acceptance = self.policy.accept(
            g_star=g_star,
            h_star=h_star,
            sigma_next=sigma_next_value,
            geometry=self.geometry,
        )

        for name, value in (
            ("x0_G", x0_g), ("assembled_x0_H", x0_h),
            ("G_star", g_star), ("H_star", h_star),
            ("accepted_G", acceptance.g), ("accepted_H", acceptance.h),
        ):
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Blueprint produced nonfinite {name} at {ordinal}.")
            self._capture(name, ordinal, value)

        invariant_error = None
        if not acceptance.terminal_release:
            invariant_error = float(
                (self.geometry.restrict(acceptance.h).float() - acceptance.g.float())
                .abs().max()
            )
            if invariant_error > self.geometry.TOLERANCE:
                raise RuntimeError(
                    f"D(H_next) != G_next at interval {ordinal}: {invariant_error}."
                )
        evaluation_id = (
            f"{self.run_id}:{ordinal}:sigma={sigma_value:.9g}:"
            f"next={sigma_next_value:.9g}"
        )
        next_state = BlueprintState(
            acceptance.g,
            acceptance.h,
            sigma_next_value,
            ordinal + 1,
            evaluation_id,
        )
        work = self.adapter.describe_work(
            global_shape=tuple(state.g.shape), crops=regions
        )
        self.telemetry.append({
            "event": "accepted_interval",
            "ordinal": ordinal,
            "evaluation_id": evaluation_id,
            "sigma": sigma_value,
            "sigma_next": sigma_next_value,
            "terminal_release": acceptance.terminal_release,
            "model_predictions": work.model_predictions,
            "global_tokens": work.global_tokens,
            "local_tokens": work.local_tokens,
            "coverage_min": float(coverage.min()),
            "coverage_max": float(coverage.max()),
            "projection_rms": acceptance.projection_rms,
            "invariant_max_abs": invariant_error,
            "accepted_H": _summary(next_state.h),
            "accepted_G": _summary(next_state.g),
        })
        return next_state, x0_h


class BlueprintEulerSampler(comfy.samplers.Sampler):
    def __init__(self, *, capture: TensorCapture | None = None) -> None:
        self.capture = capture
        self.last_telemetry: tuple[dict[str, Any], ...] = ()

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
        if denoise_mask is not None:
            raise ValueError("Blueprint first slice does not support masks.")
        if latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Blueprint first slice supports only empty-latent T2I.")
        if getattr(noise, "is_nested", False) or getattr(latent_image, "is_nested", False):
            raise ValueError("Blueprint first slice does not support nested latents.")
        validate_schedule(sigmas)
        if not self.max_denoise(model, sigmas):
            raise ValueError(
                "Blueprint first slice requires a full-denoise schedule starting "
                "at the model's maximum sigma; partial denoise is unsupported."
            )
        model_sampling = model.inner_model.model_sampling
        if "CONST" not in {item.__name__ for item in type(model_sampling).__mro__}:
            raise ValueError("Blueprint first slice requires CONST flow sampling.")
        h = model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model, sigmas)
        )
        coordinator = BlueprintCoordinator(capture=self.capture)
        state = coordinator.initialize(h, sigmas[0])
        total = len(sigmas) - 1
        for ordinal in range(total):
            if state.ordinal != ordinal:
                raise RuntimeError("Blueprint accepted-state ordinal drifted.")
            state, x0_h = coordinator.evaluate(
                guider=model,
                state=state,
                sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            )
            if callback is not None:
                callback(ordinal, x0_h, state.h, total)
        if state.ordinal != total:
            raise RuntimeError("Blueprint did not atomically accept every interval.")
        self.last_telemetry = tuple(coordinator.telemetry)
        return model_sampling.inverse_noise_scaling(sigmas[-1], state.h)
