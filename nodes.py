import torch

import comfy.model_management

from .sampling.euler import BlueprintEulerSampler
from .terminal_resampling import (
    TerminalResamplingGeometry,
    TerminalResamplingProcedure,
    validate_terminal_schedule,
)
from .configurable_resampling import (
    ConfigurableResamplingGeometry,
    ConfigurableResamplingProcedure,
)


class BlueprintCandidate3EulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Fail-closed variable-step Candidate-3 Euler sampler for the qualified "
        "native FLUX.2 Klein CFG-1 full-denoise T2I contract and compatible geometry. "
        "Use with SamplerCustomAdvanced."
    )

    def build(self):
        return (BlueprintEulerSampler(),)


class BlueprintTerminalResampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "destination": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    DESCRIPTION = (
        "Qualified finite-profile terminal-resampling Blueprint pipeline: native "
        "FLUX.2 Klein 4B BasicGuider, fixed four-step bounded Blueprint, then one "
        "fixed sigma-0.25 streamed native-local refinement pass."
    )

    def sample(self, guider, sigmas, noise_seed, destination):
        import latent_preview

        latent = destination.copy()
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("Blueprint Terminal Resampling requires a LATENT samples tensor.")
        if "noise_mask" in latent:
            raise ValueError("Blueprint Terminal Resampling does not support masks.")
        if samples.ndim != 4 or samples.shape[0] != 1 or samples.shape[1] != 128:
            raise ValueError("Blueprint Terminal Resampling requires a batch-one 128-channel latent.")
        geometry = TerminalResamplingGeometry.for_destination(tuple(samples.shape[-2:]))
        if getattr(samples, "is_nested", False) or bool(torch.count_nonzero(samples)):
            raise ValueError("Blueprint Terminal Resampling requires empty batch-one T2I latent input.")
        if type(guider).__module__ != "comfy_extras.nodes_custom_sampler" or type(guider).__name__ != "Guider_Basic":
            raise ValueError("Blueprint Terminal Resampling requires ComfyUI BasicGuider.")
        if float(getattr(guider, "cfg", float("nan"))) != 1.0:
            raise ValueError("Blueprint Terminal Resampling requires CFG exactly 1.0.")
        original_conds = getattr(guider, "original_conds", {})
        if set(original_conds) != {"positive"} or len(original_conds.get("positive", ())) != 1:
            raise ValueError("Blueprint Terminal Resampling requires one positive conditioning branch.")
        validate_terminal_schedule(sigmas)
        procedure = TerminalResamplingProcedure(seed=noise_seed, geometry=geometry)
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, 5, x0_output)
        procedure_noise = torch.zeros_like(samples, device="cpu")
        result = guider.sample(
            procedure_noise,
            samples,
            procedure,
            sigmas,
            denoise_mask=None,
            callback=callback,
            disable_pbar=False,
            seed=noise_seed,
        )
        result = result.to(comfy.model_management.intermediate_device())
        output = latent.copy()
        output.pop("noise_mask", None)
        output["samples"] = result
        output["blueprint_terminal_resampling_telemetry"] = procedure.telemetry
        denoised = output.copy()
        return output, denoised


class BlueprintConfigurablePrototype:
    @classmethod
    def INPUT_TYPES(cls):
        dimension = {"default": 32, "min": 16, "max": 512, "step": 1}
        bounded = {"default": 64, "min": 16, "max": 64, "step": 1}
        return {"required": {
            "guider": ("GUIDER",), "sigmas": ("SIGMAS",),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                     "control_after_generate": True}),
            "destination": ("LATENT",),
            "blueprint_width": ("INT", {**bounded, "default": 64}),
            "blueprint_height": ("INT", {**bounded, "default": 32}),
            "footprint_width": ("INT", {**dimension, "default": 32}),
            "footprint_height": ("INT", {**dimension, "default": 32}),
            "stride_x": ("INT", {"default": 24, "min": 1, "max": 512, "step": 1}),
            "stride_y": ("INT", {"default": 24, "min": 1, "max": 512, "step": 1}),
            "working_width": ("INT", {**bounded, "default": 64, "min": 32}),
            "working_height": ("INT", {**bounded, "default": 64, "min": 32}),
            "refinement_sigma": ("FLOAT", {"default": 0.25, "min": 0.10, "max": 0.50,
                                              "step": 0.01, "round": 0.001}),
        }}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    DESCRIPTION = (
        "Configurable bounded Blueprint prototype: fixed qualified four-step G, "
        "terminal denoised handoff, one sigma-to-zero local interval, bounded "
        "bilinear transfer, deterministic streamed overlap assembly. Native "
        "FLUX.2 Klein 4B BasicGuider/CFG-1 T2I only. Dimensions are latent cells."
    )

    def sample(self, guider, sigmas, noise_seed, destination, blueprint_width,
               blueprint_height, footprint_width, footprint_height, stride_x,
               stride_y, working_width, working_height, refinement_sigma):
        import latent_preview

        latent = destination.copy()
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
            raise ValueError("Blueprint Configurable Prototype requires a LATENT samples tensor.")
        if samples.shape[0] != 1 or samples.shape[1] != 128 or "noise_mask" in latent:
            raise ValueError("Blueprint Configurable Prototype requires unmasked batch-one 128-channel T2I.")
        geometry = ConfigurableResamplingGeometry(
            blueprint_hw=(blueprint_height, blueprint_width),
            destination_hw=tuple(samples.shape[-2:]),
            footprint_hw=(footprint_height, footprint_width),
            stride_hw=(stride_y, stride_x), working_hw=(working_height, working_width),
        )
        geometry.validate()
        if type(guider).__module__ != "comfy_extras.nodes_custom_sampler" or type(guider).__name__ != "Guider_Basic":
            raise ValueError("Blueprint Configurable Prototype requires ComfyUI BasicGuider.")
        if float(getattr(guider, "cfg", float("nan"))) != 1.0:
            raise ValueError("Blueprint Configurable Prototype requires CFG exactly 1.0.")
        validate_terminal_schedule(sigmas)
        procedure = ConfigurableResamplingProcedure(
            seed=noise_seed, geometry=geometry, refinement_sigma=refinement_sigma,
        )
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, 5, x0_output)
        result = guider.sample(torch.zeros_like(samples, device="cpu"), samples, procedure,
                               sigmas, denoise_mask=None, callback=callback,
                               disable_pbar=False, seed=noise_seed)
        output = latent.copy()
        output.pop("noise_mask", None)
        output["samples"] = result.to(comfy.model_management.intermediate_device())
        output["blueprint_configurable_telemetry"] = procedure.telemetry
        return output, output.copy()


NODE_CLASS_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": BlueprintCandidate3EulerSampler,
    "BlueprintTerminalResampling": BlueprintTerminalResampling,
    "BlueprintConfigurablePrototype": BlueprintConfigurablePrototype,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": "Blueprint Candidate-3 Euler Sampler",
    "BlueprintTerminalResampling": "Blueprint Terminal Resampling",
    "BlueprintConfigurablePrototype": "Blueprint Configurable Prototype",
}
