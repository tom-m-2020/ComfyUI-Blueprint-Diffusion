from __future__ import annotations

from typing import Any

import torch

from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock


class Flux2TerminalResamplingAdapter:
    """Strict ordinary-call adapter for the qualified terminal-resampling slice."""

    @staticmethod
    def _validate_options(model_options: dict[str, Any]) -> None:
        transformer = model_options.get("transformer_options", {})
        unsupported = {
            name
            for name in (
                "patches",
                "patches_replace",
                "wrappers",
                "callbacks",
                "reference_image_num_tokens",
                "control",
            )
            if transformer.get(name)
        }
        if unsupported:
            raise ValueError(
                "Blueprint Terminal Resampling does not support transformer "
                f"patches/wrappers: {sorted(unsupported)}."
            )
        if model_options.get("context_handler") is not None:
            raise ValueError("Blueprint Terminal Resampling does not support context wrappers.")

    @staticmethod
    def _validate_conditioning(guider) -> None:
        if type(guider).__module__ != "comfy_extras.nodes_custom_sampler" or type(guider).__name__ != "Guider_Basic":
            raise ValueError("Blueprint Terminal Resampling requires ComfyUI BasicGuider.")
        if float(getattr(guider, "cfg", float("nan"))) != 1.0:
            raise ValueError("Blueprint Terminal Resampling requires CFG exactly 1.0.")
        conds = getattr(guider, "conds", {})
        if set(conds) != {"positive"} or len(conds.get("positive", ())) != 1:
            raise ValueError(
                "Blueprint Terminal Resampling requires exactly one prepared positive branch."
            )
        unsupported_entry = {"area", "mask", "control", "gligen"}
        unsupported_model = {
            "c_concat", "concat_latent_image", "concat_mask",
            "reference_latents", "ref_latents",
        }
        for entry in conds["positive"]:
            present = unsupported_entry.intersection(entry)
            if present or entry.get("hooks") is not None:
                raise ValueError("Blueprint Terminal Resampling received spatial/hooked conditioning.")
            if unsupported_model.intersection(entry.get("model_conds", {})):
                raise ValueError("Blueprint Terminal Resampling received reference/edit conditioning.")

    def validate_prepared(
        self, *, guider, model_options, destination, model_sampling
    ) -> None:
        self._validate_conditioning(guider)
        self._validate_options(model_options)
        if tuple(destination.shape) != (1, 128, 128, 256):
            raise ValueError("Blueprint Terminal Resampling requires [1,128,128,256].")
        if "CONST" not in {item.__name__ for item in type(model_sampling).__mro__}:
            raise ValueError("Blueprint Terminal Resampling requires CONST flow sampling.")
        if float(getattr(model_sampling, "noise_scale", 1.0)) != 1.0:
            raise ValueError("Blueprint Terminal Resampling requires CONST noise_scale exactly 1.")
        base = getattr(guider, "inner_model", None)
        diffusion = getattr(base, "diffusion_model", None)
        if type(base).__module__ != "comfy.model_base" or type(base).__name__ != "Flux2":
            raise ValueError("Blueprint Terminal Resampling requires native comfy.model_base.Flux2.")
        if type(diffusion).__module__ != "comfy.ldm.flux.model" or type(diffusion).__name__ != "Flux":
            raise ValueError("Blueprint Terminal Resampling requires native comfy.ldm.flux.model.Flux.")
        params = diffusion.params
        expected = {
            "patch_size": 1,
            "in_channels": 128,
            "out_channels": 128,
            "hidden_size": 3072,
            "num_heads": 24,
            "context_in_dim": 7680,
            "axes_dim": [32, 32, 32, 32],
            "theta": 2000,
            "guidance_embed": False,
        }
        for name, required in expected.items():
            actual = getattr(params, name, None)
            if name == "axes_dim" and actual is not None:
                actual = list(actual)
            if actual != required:
                raise ValueError(
                    f"Blueprint Terminal Resampling requires Klein 4B {name}={required}, got {actual}."
                )
        if len(diffusion.double_blocks) != 5 or len(diffusion.single_blocks) != 20:
            raise ValueError("Blueprint Terminal Resampling requires 5 double and 20 single blocks.")
        if not all(type(block) is DoubleStreamBlock for block in diffusion.double_blocks):
            raise ValueError("Blueprint Terminal Resampling requires native DoubleStreamBlock modules.")
        if not all(type(block) is SingleStreamBlock for block in diffusion.single_blocks):
            raise ValueError("Blueprint Terminal Resampling requires native SingleStreamBlock modules.")
        if tuple(diffusion.img_in.weight.shape) != (3072, 128):
            raise ValueError("Blueprint Terminal Resampling found an unsupported image projection.")
        if tuple(diffusion.txt_in.weight.shape) != (3072, 7680):
            raise ValueError("Blueprint Terminal Resampling found an unsupported text projection.")

    def predict_native(
        self,
        *,
        guider,
        value: torch.Tensor,
        sigma: torch.Tensor,
        expected_hw: tuple[int, int],
        model_options: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        if tuple(value.shape) != (1, 128, *expected_hw):
            raise ValueError(
                f"Blueprint native prediction expected [1,128,{expected_hw[0]},{expected_hw[1]}], "
                f"got {tuple(value.shape)}."
            )
        options = model_options.copy()
        options["transformer_options"] = model_options.get("transformer_options", {}).copy()
        options["transformer_options"].pop("rope_options", None)
        return guider(value, sigma.expand(1), model_options=options, seed=seed)

