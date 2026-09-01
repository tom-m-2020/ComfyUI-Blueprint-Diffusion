from .sampling.euler import BlueprintEulerSampler


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


NODE_CLASS_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": BlueprintCandidate3EulerSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueprintCandidate3EulerSampler": "Blueprint Candidate-3 Euler Sampler",
}
