import torch

import comfy.model_management

from .sampling.euler import BlueprintEulerSampler
from .terminal_resampling import TerminalResamplingProcedure, validate_terminal_schedule


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
        "Exact first-slice terminal-resampling Blueprint pipeline: native FLUX.2 "
        "Klein 4B BasicGuider, fixed four-step 32x64 Blueprint, then one fixed "
        "sigma-0.25 streamed 55-region native-local refinement into 128x256."
    )

    def sample(self, guider, sigmas, noise_seed, destination):
        import latent_preview

        latent = destination.copy()
        samples = latent.get("samples")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("Blueprint Terminal Resampling requires a LATENT samples tensor.")
        if "noise_mask" in latent:
            raise ValueError("Blueprint Terminal Resampling does not support masks.")
        if tuple(samples.shape) != (1, 128, 128, 256):
            raise ValueError("Blueprint Terminal Resampling requires destination [1,128,128,256].")
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
        procedure = TerminalResamplingProcedure(seed=noise_seed)
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


NODE_CLASS_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": BlueprintCandidate3EulerSampler,
    "BlueprintTerminalResampling": BlueprintTerminalResampling,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": "Blueprint Candidate-3 Euler Sampler",
    "BlueprintTerminalResampling": "Blueprint Terminal Resampling",
}
