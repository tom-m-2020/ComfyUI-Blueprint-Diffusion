import comfy.model_management
import torch

from .configurable_resampling import (
    FLUX2_PIXEL_SCALE,
    ConfigurableResamplingGeometry,
    ConfigurableResamplingProcedure,
    geometry_from_pixels,
)
from .drift_constraints import (
    MODES,
    DriftConstrainedEuler,
    DriftConstraintNoise,
    make_policy,
)
from .sampling.euler import BlueprintEulerSampler
from .terminal_resampling import (
    TerminalResamplingGeometry,
    TerminalResamplingProcedure,
    validate_terminal_schedule,
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


class DriftConstrainedSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("LATENT",),
                "mode": (MODES,),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                       "control_after_generate": True}),
                "radius": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0,
                                      "step": 0.25}),
                "transition_bandwidth": ("FLOAT", {"default": 2.0, "min": 0.01,
                                                    "max": 64.0, "step": 0.01}),
                "downsample_factor": ("INT", {"default": 4, "min": 1, "max": 64,
                                               "step": 1}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                     "step": 0.01}),
                "normalized_threshold": ("FLOAT", {"default": 5.0 / 63.0, "min": 0.0,
                                                    "max": 2.0, "step": 0.001}),
                "calibration_start": ("FLOAT", {"default": 0.0, "min": 0.0,
                                                 "max": 1.0, "step": 0.01}),
                "calibration_end": ("FLOAT", {"default": 0.5, "min": 0.0,
                                               "max": 1.0, "step": 0.01}),
            },
            "optional": {"reference_conditioning": ("CONDITIONING",)},
        }

    RETURN_TYPES = ("DRIFT_CONSTRAINT", "NOISE")
    RETURN_NAMES = ("constraint", "noise")
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/drift"
    DESCRIPTION = (
        "Experimental stock-Klein drift policies. FSS/ILVR do not reliably preserve "
        "portrait identity; the tested FBSDiff adaptation was falsified."
    )

    def build(self, source, mode, noise_seed, radius, transition_bandwidth,
              downsample_factor, alpha, normalized_threshold, calibration_start,
              calibration_end, reference_conditioning=None):
        policy = make_policy(
            source, mode, noise_seed, radius, transition_bandwidth,
            downsample_factor, alpha, normalized_threshold, calibration_start,
            calibration_end, reference_conditioning,
        )
        return policy, DriftConstraintNoise(policy)


class DriftConstrainedEulerSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"constraint": ("DRIFT_CONSTRAINT",)}}

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Experimental CONST/Euler sampler for a paired Drift-Constrained Sampling policy. "
        "Use with native SamplerCustomAdvanced."
    )

    def build(self, constraint):
        return (DriftConstrainedEuler(constraint),)


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
            raise ValueError(  # noqa: TRY004 - preserve existing public error behavior.
                "Blueprint Terminal Resampling requires a LATENT samples tensor."
            )
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


class BlueprintDiffusion(BlueprintConfigurablePrototype):
    @classmethod
    def INPUT_TYPES(cls):
        pixel = {"min": 256, "max": 8192, "step": 16, "advanced": True}
        working = {"min": 512, "max": 1024, "step": 16, "advanced": True}
        return {"required": {
            "guider": ("GUIDER", {"tooltip": "Prepared FLUX.2 Klein 4B BasicGuider. CFG must be 1."}),
            "sigmas": ("SIGMAS", {"tooltip": "Qualified four-step whole-scene Blueprint schedule."}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                     "control_after_generate": True,
                                     "tooltip": "Deterministically derives the Blueprint and every regional noise field."}),
            "destination": ("LATENT", {"tooltip": "Empty FLUX.2 destination latent. Its pixel width and height define H automatically."}),
            "geometry_mode": (["auto", "manual"], {
                "tooltip": "Auto selects only a live-qualified destination profile. Manual uses the pixel controls below."
            }),
            "blueprint_width": ("INT", {**pixel, "default": 720, "max": 1024,
                "tooltip": "G width in pixels: bounded whole-scene planning canvas. Used only in manual mode."}),
            "blueprint_height": ("INT", {**pixel, "default": 720, "max": 1024,
                "tooltip": "G height in pixels: bounded whole-scene planning canvas. Used only in manual mode."}),
            "tile_width": ("INT", {**pixel, "default": 512,
                "tooltip": "F width in pixels: portion written into the final destination by each local result."}),
            "tile_height": ("INT", {**pixel, "default": 512,
                "tooltip": "F height in pixels: portion written into the final destination by each local result."}),
            "tile_overlap_x": ("INT", {"default": 256, "min": 0, "max": 8176, "step": 16,
                "advanced": True, "tooltip": "Shared destination pixels between neighboring footprints on X; stride is tile width minus overlap."}),
            "tile_overlap_y": ("INT", {"default": 256, "min": 0, "max": 8176, "step": 16,
                "advanced": True, "tooltip": "Shared destination pixels between neighboring footprints on Y; stride is tile height minus overlap."}),
            "working_width": ("INT", {**working, "default": 1024,
                "tooltip": "W width in pixels: bounded local canvas seen by the diffusion model; must be an integer enlargement of F."}),
            "working_height": ("INT", {**working, "default": 1024,
                "tooltip": "W height in pixels: bounded local canvas seen by the diffusion model; must be an integer enlargement of F."}),
            "refinement_sigma": ("FLOAT", {"default": 0.25, "min": 0.10, "max": 0.50,
                "step": 0.01, "round": 0.001,
                "tooltip": "Late-noise strength for the single local [sigma, 0] refinement interval."}),
        }}

    DESCRIPTION = (
        "Blueprint Diffusion terminal-refinement release: G is the bounded whole-scene planning canvas; "
        "F is the destination footprint written by one tile; W is the bounded local canvas seen by the "
        "diffusion model; overlap is the shared destination area between footprints. Pixel controls use "
        "FLUX.2's 16x latent scale. Auto mode selects only qualified profiles."
    )

    def sample(self, guider, sigmas, noise_seed, destination, geometry_mode,
               blueprint_width, blueprint_height, tile_width, tile_height,
               tile_overlap_x, tile_overlap_y, working_width, working_height,
               refinement_sigma):
        samples = destination.get("samples") if isinstance(destination, dict) else None
        if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
            raise ValueError("Blueprint Diffusion requires a LATENT samples tensor to derive destination H.")
        geometry = geometry_from_pixels(
            tuple(samples.shape[-2:]), geometry_mode,
            blueprint_width=blueprint_width, blueprint_height=blueprint_height,
            tile_width=tile_width, tile_height=tile_height,
            tile_overlap_x=tile_overlap_x, tile_overlap_y=tile_overlap_y,
            working_width=working_width, working_height=working_height,
        )
        output, denoised = super().sample(
            guider, sigmas, noise_seed, destination,
            geometry.blueprint_hw[1], geometry.blueprint_hw[0],
            geometry.footprint_hw[1], geometry.footprint_hw[0],
            geometry.stride_hw[1], geometry.stride_hw[0],
            geometry.working_hw[1], geometry.working_hw[0], refinement_sigma,
        )
        telemetry = output["blueprint_configurable_telemetry"]
        telemetry["user_geometry_mode"] = geometry_mode
        telemetry["pixel_scale"] = FLUX2_PIXEL_SCALE
        telemetry["resolved_pixels"] = {
            "destination": {"width": geometry.destination_hw[1] * FLUX2_PIXEL_SCALE,
                            "height": geometry.destination_hw[0] * FLUX2_PIXEL_SCALE},
            "blueprint": {"width": geometry.blueprint_hw[1] * FLUX2_PIXEL_SCALE,
                          "height": geometry.blueprint_hw[0] * FLUX2_PIXEL_SCALE},
            "footprint": {"width": geometry.footprint_hw[1] * FLUX2_PIXEL_SCALE,
                          "height": geometry.footprint_hw[0] * FLUX2_PIXEL_SCALE},
            "overlap": {"x": (geometry.footprint_hw[1] - geometry.stride_hw[1]) * FLUX2_PIXEL_SCALE,
                        "y": (geometry.footprint_hw[0] - geometry.stride_hw[0]) * FLUX2_PIXEL_SCALE},
            "working": {"width": geometry.working_hw[1] * FLUX2_PIXEL_SCALE,
                        "height": geometry.working_hw[0] * FLUX2_PIXEL_SCALE},
        }
        return output, denoised


NODE_CLASS_MAPPINGS = {
    "DriftConstrainedSampling": DriftConstrainedSampling,
    "DriftConstrainedEulerSampler": DriftConstrainedEulerSampler,
    "BlueprintCandidate3EulerSampler": BlueprintCandidate3EulerSampler,
    "BlueprintTerminalResampling": BlueprintTerminalResampling,
    "BlueprintConfigurablePrototype": BlueprintConfigurablePrototype,
    "BlueprintDiffusion": BlueprintDiffusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DriftConstrainedSampling": "Drift-Constrained Sampling",
    "DriftConstrainedEulerSampler": "Drift-Constrained Euler Sampler",
    "BlueprintCandidate3EulerSampler": "Blueprint Candidate-3 Euler Sampler",
    "BlueprintTerminalResampling": "Blueprint Terminal Resampling",
    "BlueprintConfigurablePrototype": "Blueprint Configurable Prototype",
    "BlueprintDiffusion": "Blueprint Diffusion (Terminal Refine)",
}
