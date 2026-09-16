from __future__ import annotations

import hashlib
import math
import threading
import weakref
from dataclasses import dataclass, field
from itertools import pairwise

import torch
import torch.nn.functional as F

MODES = ("none", "fss", "ilvr", "fss_ilvr", "fbsdiff")


def tensor_fingerprint(value: torch.Tensor) -> str:
    work = value.detach().contiguous().cpu()
    header = f"{tuple(work.shape)}|{work.dtype}|".encode()
    return hashlib.sha256(header + work.numpy().tobytes()).hexdigest()


def canonical_endpoint_fingerprint(value: torch.Tensor) -> str:
    return tensor_fingerprint(value.detach().float().cpu())


@dataclass(frozen=True)
class FSSConfig:
    radius: float = 2.0
    transition_bandwidth: float = 2.0


@dataclass(frozen=True)
class ILVRConfig:
    downsample_factor: int = 4
    alpha: float = 1.0


@dataclass(frozen=True)
class FBSDiffConfig:
    normalized_threshold: float = 5.0 / 63.0
    calibration_start: float = 0.0
    calibration_end: float = 0.5
    reference_conditioning: object = field(compare=False, repr=False, default=None)


@dataclass
class EndpointProvenance:
    pair_id: str
    fingerprint: str | None = None
    generation_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, endpoint: torch.Tensor) -> None:
        with self.lock:
            self.fingerprint = canonical_endpoint_fingerprint(endpoint)
            self.generation_count += 1


@dataclass(frozen=True)
class DriftConstraintPolicy:
    mode: str
    source: torch.Tensor = field(compare=False, repr=False)
    source_fingerprint: str
    processed_source_fingerprint: str
    raw_source: torch.Tensor = field(compare=False, repr=False)
    raw_source_fingerprint: str
    model_contract: ModelContract
    model_ref: object = field(compare=False, repr=False)
    batch_index: tuple[int, ...] | None
    noise_seed: int
    pair_id: str
    provenance: EndpointProvenance = field(compare=False, repr=False)
    fss: FSSConfig | None = None
    ilvr: ILVRConfig | None = None
    fbsdiff: FBSDiffConfig | None = None


@dataclass(frozen=True)
class ModelContract:
    family: str
    model_class: str
    config_class: str
    latent_format_class: str
    latent_channels: int
    latent_dimensions: int
    sampling_class: str
    sampling_object_id: int
    sampling_parameters: tuple[tuple[str, float], ...]


def validate_source_tensor(source: object) -> torch.Tensor:
    if not isinstance(source, torch.Tensor) or source.ndim not in {4, 5}:
        raise ValueError(
            "Drift-Constrained Sampling requires a non-nested [B,C,H,W] or "
            "[B,C,T,H,W] source LATENT tensor."
        )
    if getattr(source, "is_nested", False):
        raise ValueError("Drift-Constrained Sampling does not support nested latents.")
    if not bool(torch.isfinite(source).all()):
        raise ValueError("Drift-Constrained Sampling source must be finite.")
    return source


def _sampling_parameters(model_sampling: object) -> tuple[tuple[str, float], ...]:
    values = []
    for name in ("shift", "multiplier", "noise_scale"):
        if hasattr(model_sampling, name):
            values.append((name, float(getattr(model_sampling, name))))
    return tuple(values)


def inspect_model_contract(model: object) -> ModelContract:
    if not hasattr(model, "get_model_object") or not hasattr(model, "model"):
        raise TypeError("Drift-Constrained Sampling requires a native ComfyUI MODEL.")
    latent_format = model.get_model_object("latent_format")
    model_sampling = model.get_model_object("model_sampling")
    base_model = model.model
    config = getattr(base_model, "model_config", None)
    config_class = type(config).__name__
    model_class = type(base_model).__name__
    latent_class = type(latent_format).__name__
    channels = int(getattr(latent_format, "latent_channels", 0))
    dimensions = int(getattr(latent_format, "latent_dimensions", 0))

    if model_class == "Flux2" and config_class == "Flux2" and latent_class == "Flux2":
        family = "flux2_klein"
    elif model_class == "Lumina2" and config_class == "ZImage" and latent_class == "Flux":
        family = "z_image"
    elif model_class == "Anima" and config_class == "Anima" and latent_class == "Wan21":
        family = "anima"
    else:
        raise TypeError(
            "FSS supports only qualified FLUX.2 Klein, stock Z-Image, and stock Anima "
            f"image contracts; received model={model_class}, config={config_class}, "
            f"latent_format={latent_class}."
        )

    expected = {
        "flux2_klein": (128, 2),
        "z_image": (16, 2),
        "anima": (16, 3),
    }[family]
    if (channels, dimensions) != expected:
        raise ValueError(
            f"Unexpected {family} latent contract: channels/dimensions "
            f"{channels}/{dimensions}, expected {expected[0]}/{expected[1]}."
        )
    sampling_name = ".".join(
        f"{base.__module__}.{base.__qualname__}" for base in type(model_sampling).__mro__[:-1]
    )
    return ModelContract(
        family=family,
        model_class=model_class,
        config_class=config_class,
        latent_format_class=latent_class,
        latent_channels=channels,
        latent_dimensions=dimensions,
        sampling_class=sampling_name,
        sampling_object_id=id(model_sampling),
        sampling_parameters=_sampling_parameters(model_sampling),
    )


def validate_model_contract(model: object, expected: ModelContract) -> None:
    actual = inspect_model_contract(model)
    if actual != expected:
        raise ValueError("The MODEL or its latent/sampling contract changed after policy construction.")


def canonicalize_source(
    model: object,
    source_latent: dict,
    contract: ModelContract,
    require_nonzero: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    import comfy.sample

    raw = validate_source_tensor(source_latent.get("samples"))
    is_nonzero = bool(torch.count_nonzero(raw))
    if require_nonzero and not is_nonzero:
        raise ValueError("FSS requires a real nonzero VAE-encoded source LATENT.")
    fixed = comfy.sample.fix_empty_latent_channels(
        model,
        raw.detach().clone(),
        source_latent.get("downscale_ratio_spacial"),
        source_latent.get("downscale_ratio_temporal"),
    )
    fixed = validate_source_tensor(fixed)
    expected_rank = 5 if contract.family == "anima" else 4
    if fixed.ndim != expected_rank or fixed.shape[1] != contract.latent_channels:
        raise ValueError(
            f"Canonical {contract.family} source must have rank {expected_rank} and "
            f"{contract.latent_channels} channels; got {tuple(fixed.shape)}."
        )
    if contract.family == "anima" and fixed.shape[2] != 1:
        raise ValueError("Initial Anima FSS support is image-only and requires T=1.")
    processed = (
        model.model.process_latent_in(fixed.detach().clone())
        if is_nonzero else fixed.detach().clone()
    )
    if not isinstance(processed, torch.Tensor) or processed.shape != fixed.shape:
        raise ValueError("MODEL process_latent_in changed the canonical source shape.")
    if not bool(torch.isfinite(processed).all()):
        raise ValueError("MODEL process_latent_in produced a non-finite source.")
    return fixed.detach().clone(), processed.detach().clone()


def validate_reference_conditioning(conditioning: object) -> None:
    if not isinstance(conditioning, list) or not conditioning:
        raise ValueError("fbsdiff mode requires explicit reference_conditioning.")
    prohibited = {"control", "gligen", "hooks", "mask", "set_area_to_bounds"}
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError("Unsupported FBSDiff reference conditioning structure.")
        found = prohibited.intersection(item[1])
        if found:
            names = ", ".join(sorted(found))
            raise ValueError(f"FBSDiff reference conditioning contains unsupported stateful fields: {names}.")


def make_policy(
    model: object,
    source_latent: dict,
    mode: str,
    noise_seed: int,
    radius: float,
    transition_bandwidth: float,
    downsample_factor: int,
    alpha: float,
    normalized_threshold: float,
    calibration_start: float,
    calibration_end: float,
    reference_conditioning: object = None,
) -> DriftConstraintPolicy:
    if mode not in MODES:
        raise ValueError(f"Unsupported drift mode {mode!r}.")
    if not isinstance(source_latent, dict):
        raise TypeError("source must be a LATENT dictionary.")
    if "noise_mask" in source_latent:
        raise ValueError("Drift-Constrained Sampling does not support latent noise masks.")
    contract = inspect_model_contract(model)
    if contract.family != "flux2_klein" and mode not in {"none", "fss"}:
        raise TypeError(
            f"Mode {mode!r} remains FLUX.2 Klein-only; {contract.family} supports only "
            "native Gaussian or FSS NOISE."
        )
    raw_source, source = canonicalize_source(
        model, source_latent, contract, require_nonzero=mode in {"fss", "fss_ilvr"}
    )
    validate_model_contract(model, contract)
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("FSS radius must be finite and nonnegative.")
    if transition_bandwidth <= 0 or not math.isfinite(transition_bandwidth):
        raise ValueError("FSS transition_bandwidth must be finite and positive.")
    if downsample_factor < 1:
        raise ValueError("ILVR downsample_factor must be positive.")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("ILVR alpha must be finite and nonnegative.")
    if not 0 <= normalized_threshold <= 2:
        raise ValueError("FBSDiff normalized_threshold must be in [0,2].")
    if not 0 <= calibration_start < calibration_end <= 1:
        raise ValueError("FBSDiff calibration must satisfy 0 <= start < end <= 1.")
    if mode == "fbsdiff":
        validate_reference_conditioning(reference_conditioning)

    batch_index = source_latent.get("batch_index")
    if batch_index is not None:
        batch_index = tuple(int(value) for value in batch_index)
        if len(batch_index) != source.shape[0]:
            raise ValueError("source batch_index length must equal source batch size.")
    owned = source.detach().clone()
    raw_owned = raw_source.detach().clone()
    source_hash = tensor_fingerprint(owned)
    raw_source_hash = tensor_fingerprint(raw_owned)
    pair_id = hashlib.sha256(
        f"{source_hash}|{raw_source_hash}|{contract}|{noise_seed}|{mode}|"
        f"{radius}|{transition_bandwidth}|{batch_index}".encode()
    ).hexdigest()
    provenance = EndpointProvenance(pair_id)
    return DriftConstraintPolicy(
        mode=mode,
        source=owned,
        source_fingerprint=source_hash,
        processed_source_fingerprint=canonical_endpoint_fingerprint(owned),
        raw_source=raw_owned,
        raw_source_fingerprint=raw_source_hash,
        model_contract=contract,
        model_ref=weakref.ref(model),
        batch_index=batch_index,
        noise_seed=int(noise_seed),
        pair_id=pair_id,
        provenance=provenance,
        fss=FSSConfig(radius, transition_bandwidth) if mode in {"fss", "fss_ilvr"} else None,
        ilvr=ILVRConfig(downsample_factor, alpha) if mode in {"ilvr", "fss_ilvr"} else None,
        fbsdiff=FBSDiffConfig(
            normalized_threshold, calibration_start, calibration_end, reference_conditioning
        ) if mode == "fbsdiff" else None,
    )


def frequency_mask(height: int, width: int, cutoff: float, bandwidth: float) -> torch.Tensor:
    fy = torch.arange(height, dtype=torch.float32) - height // 2
    fx = torch.arange(width, dtype=torch.float32) - width // 2
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    outside = torch.exp(-torch.clamp(radius - cutoff, min=0).square() / (2 * bandwidth**2))
    return torch.where(radius <= cutoff, torch.ones_like(radius), outside)


def structured_noise(source: torch.Tensor, gaussian: torch.Tensor, config: FSSConfig) -> torch.Tensor:
    if source.shape != gaussian.shape:
        raise ValueError("FSS source and Gaussian endpoint shapes must match exactly.")
    source_fft = torch.fft.fft2(source.float(), dim=(-2, -1))
    gaussian_fft = torch.fft.fft2(gaussian.float(), dim=(-2, -1))
    source_phase = torch.angle(source_fft)
    gaussian_phase = torch.angle(gaussian_fft)
    shifted = frequency_mask(
        source.shape[-2], source.shape[-1], config.radius, config.transition_bandwidth
    )
    mask = torch.fft.ifftshift(shifted).to(source_phase)
    phase = source_phase * mask + gaussian_phase * (1.0 - mask)
    spectrum = gaussian_fft.abs() * torch.exp(1j * phase)
    return torch.fft.ifft2(spectrum, dim=(-2, -1)).real.to(gaussian.dtype)


def lowpass(value: torch.Tensor, factor: int) -> torch.Tensor:
    height, width = value.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"ILVR factor {factor} must divide latent geometry {height}x{width}.")
    coarse = F.interpolate(value, (height // factor, width // factor), mode="area")
    return F.interpolate(coarse, (height, width), mode="nearest")


def ilvr_correct(
    proposal: torch.Tensor, source_at_sigma_next: torch.Tensor, config: ILVRConfig
) -> torch.Tensor:
    mismatch = lowpass(source_at_sigma_next, config.downsample_factor) - lowpass(
        proposal, config.downsample_factor
    )
    return proposal + config.alpha * mismatch


def dct_matrix(size: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(math.pi / size * frequencies * positions)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def dct_2d(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(matrix, value), matrix.t())


def idct_2d(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(matrix.t(), value), matrix)


def fbsdiff_active(config: FBSDiffConfig, ordinal: int, total_steps: int) -> bool:
    position = ordinal / total_steps
    return config.calibration_start <= position < config.calibration_end


def fbsdiff_substitute(
    target: torch.Tensor, reference: torch.Tensor, config: FBSDiffConfig
) -> torch.Tensor:
    if target.shape != reference.shape or target.ndim != 4:
        raise ValueError("FBSDiff reference and target states must have identical [B,C,H,W] shapes.")
    height, width = target.shape[-2:]
    if height != width:
        raise ValueError("Initial FBSDiff production mode supports only square sampler latents.")
    matrix = dct_matrix(height, target.device)
    target_dct = dct_2d(target.float(), matrix)
    reference_dct = dct_2d(reference.float(), matrix)
    rows = torch.arange(height, device=target.device).unsqueeze(1)
    columns = torch.arange(width, device=target.device).unsqueeze(0)
    mask = ((rows + columns) / (height - 1) > config.normalized_threshold).to(target_dct.dtype)
    merged = reference_dct * mask + target_dct * (1.0 - mask)
    return idct_2d(merged, matrix).to(target.dtype)


class DriftConstraintNoise:
    def __init__(self, policy: DriftConstraintPolicy):
        self.policy = policy
        self.seed = policy.noise_seed

    def generate_noise(self, input_latent: dict) -> torch.Tensor:
        import comfy.sample

        if not isinstance(input_latent, dict) or "noise_mask" in input_latent:
            raise ValueError("Drift NOISE requires the original unmasked source LATENT.")
        model = self.policy.model_ref()
        if model is None:
            raise ValueError("The MODEL used to construct this drift policy is no longer available.")
        validate_model_contract(model, self.policy.model_contract)
        if tensor_fingerprint(self.policy.source) != self.policy.source_fingerprint:
            raise ValueError("Drift policy source snapshot was mutated after construction.")
        if tensor_fingerprint(self.policy.raw_source) != self.policy.raw_source_fingerprint:
            raise ValueError("Drift policy raw source snapshot was mutated after construction.")
        source = validate_source_tensor(input_latent.get("samples"))
        if tensor_fingerprint(source) != self.policy.raw_source_fingerprint:
            raise ValueError("Drift NOISE source differs from the policy source snapshot.")
        batch_index = input_latent.get("batch_index")
        actual_index = None if batch_index is None else tuple(int(value) for value in batch_index)
        if actual_index != self.policy.batch_index:
            raise ValueError("Drift NOISE batch_index differs from the policy batch_index.")
        gaussian = comfy.sample.prepare_noise(source, self.seed, batch_index)
        endpoint = (
            structured_noise(self.policy.source.to(gaussian), gaussian, self.policy.fss)
            if self.policy.fss is not None else gaussian
        )
        self.policy.provenance.record(endpoint)
        return endpoint


def validate_sigmas(sigmas: torch.Tensor) -> None:
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise ValueError("Drift sampler requires a one-dimensional SIGMAS tensor with an interval.")
    if not bool(torch.isfinite(sigmas).all()) or not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("Drift sampler requires finite, strictly decreasing sigmas.")
    if float(sigmas[-1]) != 0.0 or bool((sigmas[:-1] <= 0).any()):
        raise ValueError("Drift sampler requires positive evaluation sigmas ending exactly at zero.")


class DriftConstrainedEuler:
    def __init__(self, policy: DriftConstraintPolicy):
        self.policy = policy

    @staticmethod
    def _max_denoise(model, sigmas: torch.Tensor) -> bool:
        maximum = float(model.inner_model.model_sampling.sigma_max)
        first = float(sigmas[0])
        return math.isclose(maximum, first, rel_tol=1e-5) or first > maximum

    def _validate_inputs(self, model, sigmas, noise, latent_image, denoise_mask) -> None:
        import comfy.model_sampling

        if self.policy.model_contract.family != "flux2_klein":
            raise TypeError("Drift-Constrained Euler remains FLUX.2 Klein-only.")
        if denoise_mask is not None:
            raise ValueError("Drift-Constrained Euler does not support noise masks.")
        validate_sigmas(sigmas)
        source = validate_source_tensor(latent_image)
        if noise.shape != source.shape or self.policy.source.shape != source.shape:
            raise ValueError("Drift source, target latent, and noise shapes must match exactly.")
        if canonical_endpoint_fingerprint(source) != self.policy.processed_source_fingerprint:
            raise ValueError("Sampler source differs from the policy source snapshot.")
        if tensor_fingerprint(self.policy.source) != self.policy.source_fingerprint:
            raise ValueError("Drift policy source snapshot was mutated after construction.")
        if not isinstance(model.inner_model.model_sampling, comfy.model_sampling.CONST):
            raise TypeError("Drift-Constrained Euler supports native CONST sampling only.")
        expected = self.policy.provenance.fingerprint
        if expected is None or self.policy.provenance.generation_count < 1:
            raise ValueError("Paired drift NOISE was not generated before sampler invocation.")
        if canonical_endpoint_fingerprint(noise) != expected:
            raise ValueError("Sampler noise does not match the paired drift NOISE provenance.")

    def _prepare_reference(self, model, noise, latent_image, denoise_mask, seed):
        import comfy.sampler_helpers
        import comfy.samplers

        raw = self.policy.fbsdiff.reference_conditioning
        converted = comfy.sampler_helpers.convert_cond(raw)
        reference = comfy.samplers.process_conds(
            model.inner_model,
            noise,
            {"positive": converted},
            noise.device,
            latent_image,
            denoise_mask,
            seed,
            latent_shapes=[latent_image.shape],
        )
        return reference["positive"]

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        del disable_pbar
        self._validate_inputs(model, sigmas, noise, latent_image, denoise_mask)
        sampling = model.inner_model.model_sampling
        state = sampling.noise_scaling(
            sigmas[0], noise, latent_image, self._max_denoise(model, sigmas)
        )
        reference = state.clone() if self.policy.fbsdiff is not None else None
        reference_cond = (
            self._prepare_reference(model, noise, latent_image, denoise_mask, extra_args.get("seed", 0))
            if self.policy.fbsdiff is not None else None
        )
        total_steps = len(sigmas) - 1
        for ordinal, (sigma, sigma_next) in enumerate(pairwise(sigmas)):
            if self.policy.fbsdiff is not None:
                if fbsdiff_active(self.policy.fbsdiff, ordinal, total_steps):
                    state = fbsdiff_substitute(state, reference, self.policy.fbsdiff)
                import comfy.samplers
                reference_denoised = comfy.samplers.calc_cond_batch(
                    model.inner_model,
                    [reference_cond],
                    reference,
                    sigma.expand(reference.shape[0]),
                    extra_args["model_options"],
                )[0]
            denoised = model(
                state,
                sigma.expand(state.shape[0]),
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            )
            proposal = state + (state - denoised) / sigma * (sigma_next - sigma)
            if reference is not None:
                reference = reference + (reference - reference_denoised) / sigma * (sigma_next - sigma)
            if self.policy.ilvr is not None and float(sigma_next) > 0.0:
                source_at_next = sampling.noise_scaling(
                    sigma_next, noise, latent_image, False
                )
                proposal = ilvr_correct(proposal, source_at_next, self.policy.ilvr)
            state = proposal
            if callback is not None:
                callback(ordinal, denoised, state, total_steps)
        return sampling.inverse_noise_scaling(sigmas[-1], state)
