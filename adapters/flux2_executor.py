from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from einops import rearrange

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.flux import math as flux_math
from comfy.ldm.flux.layers import DoubleStreamBlock, SingleStreamBlock, timestep_embedding

from .base import RegionPredictionSet


@dataclass
class _Invocation:
    x: torch.Tensor
    timestep: torch.Tensor
    context: torch.Tensor
    y: torch.Tensor | None
    guidance: torch.Tensor | None
    transformer_options: dict[str, Any]


@dataclass
class _State:
    img: torch.Tensor
    txt: torch.Tensor
    vec_orig: torch.Tensor
    double_vec: Any
    single_vec: Any
    pe: torch.Tensor
    options: dict[str, Any]


class _OneBlockContext:
    def __init__(self) -> None:
        self.key: tuple[str, int] | None = None
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.normal: torch.Tensor | None = None
        self.bytes = 0

    @staticmethod
    def _key(options: dict[str, Any]) -> tuple[str, int]:
        return str(options["block_type"]), int(options["block_index"])

    def capture(self, q, k, v, pe, attn_mask, extra_options):
        start, end = map(int, extra_options["img_slice"])
        if end != k.shape[2]:
            raise RuntimeError("Blueprint source generated-token slice drifted.")
        positioned = flux_math.apply_rope1(k, pe)
        self.key = self._key(extra_options)
        self.k = positioned[:, :, start:].detach()
        self.v = v[:, :, start:].detach()
        self.bytes = (
            self.k.numel() * self.k.element_size()
            + self.v.numel() * self.v.element_size()
        )
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def consume(self, q, k, v, pe, attn_mask, extra_options):
        if self.key != self._key(extra_options) or self.k is None or self.v is None:
            raise RuntimeError("Blueprint local block has no matching current source K/V.")
        self.normal = flux_math.attention(
            q, k, v, pe=pe, mask=attn_mask, transformer_options=extra_options
        )
        return {
            "q": flux_math.apply_rope1(q, pe),
            "k": torch.cat((flux_math.apply_rope1(k, pe), self.k), dim=2),
            "v": torch.cat((v, self.v), dim=2),
            "pe": None,
            "attn_mask": attn_mask,
        }

    def restore_text(self, output, extra_options):
        if self.normal is None:
            raise RuntimeError("Blueprint ordinary local text attention was not retained.")
        text_tokens = int(extra_options["img_slice"][0])
        result = output.clone()
        result[:, :text_tokens] = self.normal[:, :text_tokens]
        self.normal = None
        return result

    def release(self) -> None:
        self.key = None
        self.k = None
        self.v = None
        self.normal = None
        self.bytes = 0


class Flux2BlockExecutor:
    """Exact terminal-only native FLUX.2 block-major execution."""

    SOURCE_TOKENS = 18_432
    TEXT_TOKENS = 512

    def __init__(self, adapter) -> None:
        self.adapter = adapter

    @staticmethod
    def _interrupt() -> None:
        comfy.model_management.throw_exception_if_processing_interrupted()

    @staticmethod
    def _native(guider):
        base = getattr(guider, "inner_model", None)
        diffusion = getattr(base, "diffusion_model", None)
        if type(base).__module__ != "comfy.model_base" or type(base).__name__ != "Flux2":
            raise ValueError("Blueprint terminal context requires native comfy.model_base.Flux2.")
        if type(diffusion).__module__ != "comfy.ldm.flux.model" or type(diffusion).__name__ != "Flux":
            raise ValueError("Blueprint terminal context requires native comfy.ldm.flux.model.Flux.")
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
        for name, value in expected.items():
            actual = getattr(params, name, None)
            if name == "axes_dim":
                actual = list(actual) if actual is not None else None
            if actual != value:
                raise ValueError(
                    f"Blueprint terminal context requires Klein 4B {name}={value}, got {actual}."
                )
        if len(diffusion.double_blocks) != 5 or len(diffusion.single_blocks) != 20:
            raise ValueError("Blueprint terminal context requires exactly 5 double and 20 single blocks.")
        if not all(type(block) is DoubleStreamBlock for block in diffusion.double_blocks):
            raise ValueError("Blueprint terminal context requires native DoubleStreamBlock modules.")
        if not all(type(block) is SingleStreamBlock for block in diffusion.single_blocks):
            raise ValueError("Blueprint terminal context requires native SingleStreamBlock modules.")
        if diffusion.img_in.weight.shape != (3072, 128):
            raise ValueError("Blueprint terminal context found an unsupported image projection.")
        if diffusion.txt_in.weight.shape != (3072, 7680):
            raise ValueError("Blueprint terminal context found an unsupported text projection.")
        return base, diffusion

    @staticmethod
    def _validate_options(model_options: dict[str, Any]) -> None:
        transformer = model_options.get("transformer_options", {})
        unsupported = {
            name for name in (
                "patches", "patches_replace", "wrappers", "callbacks",
                "reference_image_num_tokens", "control",
            )
            if transformer.get(name)
        }
        if unsupported:
            raise ValueError(
                "Blueprint terminal context does not support transformer patches/wrappers: "
                f"{sorted(unsupported)}."
            )
        if model_options.get("context_handler") is not None:
            raise ValueError("Blueprint terminal context does not support context-window wrappers.")

    @staticmethod
    def _validate_conditioning(guider) -> None:
        conds = getattr(guider, "conds", {})
        positive = conds.get("positive", ())
        if len(positive) != 1:
            raise ValueError(
                "Blueprint terminal context requires exactly one prepared positive conditioning entry."
            )

    def _capture(
        self, *, guider, value, sigma, rope, model_options, seed, expected_hw
    ) -> _Invocation:
        self._interrupt()
        options = comfy.patcher_extension.copy_nested_dicts(model_options)
        transformer = options.setdefault("transformer_options", {})
        transformer["rope_options"] = dict(rope)
        captured: list[_Invocation] = []

        def wrapper(_executor, x, timestep, context, y=None, guidance=None,
                    ref_latents=None, control=None, transformer_options={}, **kwargs):
            if ref_latents is not None or control is not None:
                raise ValueError("Blueprint terminal context does not support references or ControlNet.")
            if kwargs.get("attention_mask") is not None or kwargs.get("ref_latents_method") is not None:
                raise ValueError("Blueprint terminal context received unsupported FLUX invocation options.")
            if tuple(x.shape) != (1, 128, *expected_hw):
                raise RuntimeError(f"Blueprint captured unexpected input shape {tuple(x.shape)}.")
            if context.ndim != 3 or tuple(context.shape[:2]) != (1, self.TEXT_TOKENS):
                raise ValueError(
                    f"Blueprint terminal context requires 512 prepared text tokens, got {tuple(context.shape)}."
                )
            captured.append(_Invocation(
                x=x, timestep=timestep, context=context, y=y, guidance=guidance,
                transformer_options=transformer_options.copy(),
            ))
            return torch.zeros_like(x)

        comfy.patcher_extension.add_wrapper(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            wrapper,
            options,
            is_model_options=True,
        )
        _ = guider(value, sigma.expand(1), model_options=options, seed=seed)
        if len(captured) != 1:
            raise RuntimeError(
                f"Blueprint expected one prepared native FLUX invocation, got {len(captured)}."
            )
        return captured[0]

    @staticmethod
    def _patch(options, name, callback) -> None:
        options.setdefault("patches", {}).setdefault(name, []).append(callback)

    def _prepare(self, diffusion, invocation: _Invocation, context, mode: str) -> _State:
        options = invocation.transformer_options.copy()
        options.pop("wrappers", None)
        img_tokens, img_ids = diffusion.process_img(
            invocation.x, transformer_options=options
        )
        txt_ids = torch.zeros(
            (1, invocation.context.shape[1], len(diffusion.params.axes_dim)),
            device=invocation.x.device, dtype=torch.float32,
        )
        for axis in diffusion.params.txt_ids_dims:
            txt_ids[:, :, axis] = torch.linspace(
                0, invocation.context.shape[1] - 1,
                steps=invocation.context.shape[1], device=invocation.x.device,
            )
        img = diffusion.img_in(img_tokens)
        vec = diffusion.time_in(timestep_embedding(invocation.timestep, 256).to(img.dtype))
        if diffusion.vector_in is not None:
            y = invocation.y
            if y is None:
                y = torch.zeros((1, diffusion.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + diffusion.vector_in(y[:, :diffusion.params.vec_in_dim])
        txt = invocation.context
        if diffusion.txt_norm is not None:
            txt = diffusion.txt_norm(txt)
        txt = diffusion.txt_in(txt)
        for patch in options.get("patches", {}).get("post_input", ()):
            out = patch({"img": img, "txt": txt, "img_ids": img_ids,
                         "txt_ids": txt_ids, "transformer_options": options})
            img, txt, img_ids, txt_ids = out["img"], out["txt"], out["img_ids"], out["txt_ids"]
        pe = diffusion.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        if mode == "source":
            self._patch(options, "attn1_patch", context.capture)
        else:
            self._patch(options, "attn1_patch", context.consume)
            self._patch(options, "attn1_output_patch", context.restore_text)
        double_vec = vec
        single_vec = vec
        if diffusion.params.global_modulation:
            double_vec = (
                diffusion.double_stream_modulation_img(vec),
                diffusion.double_stream_modulation_txt(vec),
            )
            single_vec, _ = diffusion.single_stream_modulation(vec)
        return _State(img, txt, vec, double_vec, single_vec, pe, options)

    @staticmethod
    def _options(state: _State, kind: str, index: int, total: int):
        options = state.options.copy()
        options.update({"total_blocks": total, "block_type": kind, "block_index": index})
        if kind == "single":
            options["img_slice"] = [state.txt.shape[1], state.img.shape[1]]
        return options

    def _double(self, diffusion, state: _State, index: int) -> None:
        state.img, state.txt = diffusion.double_blocks[index](
            img=state.img, txt=state.txt, vec=state.double_vec, pe=state.pe,
            attn_mask=None,
            transformer_options=self._options(state, "double", index, 5),
        )

    @staticmethod
    def _enter_single(state: _State) -> None:
        if state.img.dtype == torch.float16:
            state.img = torch.nan_to_num(state.img, nan=0.0, posinf=65504, neginf=-65504)
        state.img = torch.cat((state.txt, state.img), dim=1)

    def _single(self, diffusion, state: _State, index: int) -> None:
        state.img = diffusion.single_blocks[index](
            state.img, vec=state.single_vec, pe=state.pe, attn_mask=None,
            transformer_options=self._options(state, "single", index, 20),
        )

    @staticmethod
    def _final(diffusion, state: _State, latent, sigma, model_sampling):
        raw = diffusion.final_layer(state.img[:, state.txt.shape[1]:], state.vec_orig)
        h, w = latent.shape[-2:]
        output = rearrange(
            raw, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h, w=w, ph=diffusion.patch_size, pw=diffusion.patch_size,
        )[..., :h, :w]
        return model_sampling.calculate_denoised(sigma, output.float(), latent)

    def predict_regions(
        self, *, guider, g, h, sigma, canvas, regions, model_options, seed
    ) -> RegionPredictionSet:
        started = time.perf_counter()
        source_state = None
        crop_states: list[_State] = []
        predictions: list[torch.Tensor] = []
        context = _OneBlockContext()
        barrier_records = []
        try:
            self._interrupt()
            base, diffusion = self._native(guider)
            self._validate_options(model_options)
            self._validate_conditioning(guider)
            expected = tuple((y, x) for y in (0, 24, 48, 72, 96) for x in
                             (0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 224))
            if tuple((r.y, r.x) for r in regions) != expected:
                raise ValueError("Blueprint terminal context requires the qualified ordered 55-crop layout.")
            source_invocation = self._capture(
                guider=guider, value=g, sigma=sigma,
                rope={"scale_y": 127.0 / 95.0, "scale_x": 255.0 / 191.0},
                model_options=model_options, seed=seed, expected_hw=(96, 192),
            )
            crop_invocations = []
            for region in regions:
                self._interrupt()
                latent = h[:, :, region.y:region.y2, region.x:region.x2]
                crop_invocations.append(self._capture(
                    guider=guider, value=latent, sigma=sigma,
                    rope={"shift_y": float(region.y), "shift_x": float(region.x)},
                    model_options=model_options, seed=seed, expected_hw=(32, 32),
                ))
            source_state = self._prepare(diffusion, source_invocation, context, "source")
            crop_states = [self._prepare(diffusion, item, context, "local") for item in crop_invocations]
            del crop_invocations, source_invocation

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_event = end_event = None

            for kind, count in (("double", 5), ("single", 20)):
                if kind == "single":
                    self._enter_single(source_state)
                    for state in crop_states:
                        self._enter_single(state)
                for index in range(count):
                    self._interrupt()
                    if kind == "double":
                        self._double(diffusion, source_state, index)
                    else:
                        self._single(diffusion, source_state, index)
                    if context.k is None or context.k.shape[2] != self.SOURCE_TOKENS:
                        raise RuntimeError("Blueprint source K/V token count drifted.")
                    for state in crop_states:
                        self._interrupt()
                        if kind == "double":
                            self._double(diffusion, state, index)
                        else:
                            self._single(diffusion, state, index)
                    if context.normal is not None:
                        raise RuntimeError("Blueprint retained unfinished local attention state.")
                    barrier_records.append({
                        "block_type": kind,
                        "block_index": index,
                        "source_kv_bytes": context.bytes,
                        "allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
                        "reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
                    })
                    context.release()

            for region, state in zip(regions, crop_states):
                self._interrupt()
                latent = h[:, :, region.y:region.y2, region.x:region.x2]
                prediction = self._final(diffusion, state, latent, sigma, base.model_sampling)
                if tuple(prediction.shape) != tuple(latent.shape) or not bool(torch.isfinite(prediction).all()):
                    raise RuntimeError(f"Blueprint specialized crop {region.index} is invalid.")
                predictions.append(prediction)
            if len(predictions) != len(regions):
                raise RuntimeError("Blueprint specialized executor did not complete every crop.")
            if start_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                cuda_ms = float(start_event.elapsed_time(end_event))
                peak_allocated = int(torch.cuda.max_memory_allocated())
                peak_reserved = int(torch.cuda.max_memory_reserved())
            else:
                cuda_ms = 0.0
                peak_allocated = peak_reserved = 0
            return RegionPredictionSet(tuple(predictions), {
                "terminal_context_source_performed": True,
                "terminal_context_source_blocks": 25,
                "terminal_global_prediction_performed": False,
                "source_final_projection_performed": False,
                "global_state_status": "retained_preterminal_unsynchronized",
                "descriptor_capture_calls": 56,
                "source_context_tokens": self.SOURCE_TOKENS,
                "cpu_kv_cache_bytes": 0,
                "cpu_to_gpu_kv_transfer_bytes": 0,
                "specialized_cuda_ms": cuda_ms,
                "specialized_wall_seconds": time.perf_counter() - started,
                "specialized_peak_allocated_bytes": peak_allocated,
                "specialized_peak_reserved_bytes": peak_reserved,
                "block_barriers": tuple(barrier_records),
            })
        finally:
            context.release()
            predictions.clear() if len(predictions) != len(regions) else None
            crop_states.clear()
            source_state = None
