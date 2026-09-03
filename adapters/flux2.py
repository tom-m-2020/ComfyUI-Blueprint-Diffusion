from __future__ import annotations

from typing import Any

import torch

from .base import RegionPredictionSet, WorkEstimate
from ..regions import Region


class Flux2Adapter:
    family = "flux2"

    @staticmethod
    def _options(
        base: dict[str, Any], rope: dict[str, float]
    ) -> dict[str, Any]:
        options = base.copy()
        transformer = base.get("transformer_options", {}).copy()
        options["transformer_options"] = transformer
        transformer["rope_options"] = rope
        return options

    def validate_run(
        self,
        *,
        guider,
        high_shape,
        global_shape,
        crops,
        sigmas,
        latent,
    ) -> None:
        if float(getattr(guider, "cfg", float("nan"))) != 1.0:
            raise ValueError("Blueprint first slice requires CFG exactly 1.0.")
        inner_model = getattr(guider, "inner_model", None)
        if inner_model is None or inner_model.__class__.__name__ != "Flux2":
            name = type(inner_model).__name__ if inner_model is not None else "None"
            raise ValueError(f"Blueprint first slice requires native ComfyUI Flux2, got {name}.")
        diffusion_model = getattr(inner_model, "diffusion_model", None)
        diffusion_module = type(diffusion_model).__module__ if diffusion_model is not None else ""
        if (
            diffusion_model is None
            or diffusion_model.__class__.__name__ != "Flux"
            or diffusion_module != "comfy.ldm.flux.model"
        ):
            name = type(diffusion_model).__name__ if diffusion_model is not None else "None"
            raise ValueError(f"Blueprint requires the FLUX.2 coordinate backend, got {name}.")
        high_hw = tuple(high_shape[-2:])
        expected_global = tuple(value // 4 * 3 for value in high_hw)
        if any(value % 4 for value in high_hw) or tuple(global_shape[-2:]) != expected_global:
            raise ValueError(
                "Blueprint FLUX.2 adapter received incompatible high/global geometry."
            )
        if not crops:
            raise ValueError("Blueprint FLUX.2 adapter requires planned crops.")
        if latent.ndim != 4 or latent.shape[0] != 1:
            raise ValueError("Blueprint FLUX.2 adapter requires one 4-D image latent.")
        unsupported_entry_keys = {"area", "mask", "control", "gligen"}
        unsupported_model_conds = {
            "c_concat", "concat_latent_image", "concat_mask",
            "reference_latents", "ref_latents",
        }
        for branch, entries in getattr(guider, "conds", {}).items():
            for entry in entries:
                present = unsupported_entry_keys.intersection(entry)
                if present:
                    raise ValueError(
                        f"Blueprint does not support {branch} conditioning keys "
                        f"{sorted(present)}."
                    )
                if entry.get("hooks") is not None:
                    raise ValueError("Blueprint does not support conditioning hooks.")
                model_conds = entry.get("model_conds", {})
                present_model = unsupported_model_conds.intersection(model_conds)
                if present_model:
                    raise ValueError(
                        "Blueprint does not support spatial/reference model conditions "
                        f"{sorted(present_model)}."
                    )

    def predict_global(
        self, *, guider, g, sigma, canvas, model_options, seed
    ) -> torch.Tensor:
        target_h, target_w = canvas
        global_h, global_w = g.shape[-2:]
        rope = {
            "scale_y": (target_h - 1.0) / (global_h - 1.0),
            "scale_x": (target_w - 1.0) / (global_w - 1.0),
        }
        return guider(
            g,
            sigma.expand(g.shape[0]),
            model_options=self._options(model_options, rope),
            seed=seed,
        )

    def predict_region(
        self, *, guider, h_view, sigma, canvas, region, model_options, seed
    ) -> torch.Tensor:
        rope = {"shift_y": float(region.y), "shift_x": float(region.x)}
        return guider(
            h_view,
            sigma.expand(h_view.shape[0]),
            model_options=self._options(model_options, rope),
            seed=seed,
        )

    def predict_regions(
        self, *, guider, g, h, sigma, sigma_next, canvas, regions,
        model_options, seed,
    ) -> RegionPredictionSet:
        terminal = float(sigma_next) == 0.0
        qualified = (
            terminal
            and tuple(h.shape) == (1, 128, 128, 256)
            and tuple(g.shape) == (1, 128, 96, 192)
            and len(regions) == 55
            and all((r.height, r.width) == (32, 32) for r in regions)
        )
        if qualified:
            from .flux2_executor import Flux2BlockExecutor

            return Flux2BlockExecutor(self).predict_regions(
                guider=guider, g=g, h=h, sigma=sigma, canvas=canvas,
                regions=regions, model_options=model_options, seed=seed,
            )

        predictions = []
        for region in regions:
            view = h[:, :, region.y:region.y2, region.x:region.x2]
            predictions.append(self.predict_region(
                guider=guider, h_view=view, sigma=sigma, canvas=canvas,
                region=region, model_options=model_options, seed=seed,
            ))
        return RegionPredictionSet(tuple(predictions), {
            "terminal_context_source_performed": False,
            "terminal_context_source_blocks": 0,
            "terminal_global_prediction_performed": False,
            "source_final_projection_performed": False,
        })

    def describe_work(self, *, global_shape, crops) -> WorkEstimate:
        global_tokens = global_shape[-2] * global_shape[-1]
        local_tokens = sum(region.height * region.width for region in crops)
        return WorkEstimate(global_tokens, local_tokens, 1 + len(crops))
