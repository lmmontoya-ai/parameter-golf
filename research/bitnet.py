from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None


TERNARY_PACKING_VERSION = 1
TERNARY_SUPPORTED_SCALE_MODES = {"per_row_absmean", "per_tensor_absmean"}


def _validate_scale_mode(scale_mode: str) -> None:
    if scale_mode not in TERNARY_SUPPORTED_SCALE_MODES:
        raise ValueError(
            f"scale_mode must be one of {sorted(TERNARY_SUPPORTED_SCALE_MODES)}, got {scale_mode!r}"
        )


def scale_mode_to_quant_mode(scale_mode: str) -> str:
    _validate_scale_mode(scale_mode)
    return scale_mode


def qmax_for_bits(num_bits: int) -> int:
    if num_bits < 2 or num_bits > 8:
        raise ValueError(f"num_bits must be in [2, 8], got {num_bits}")
    return (1 << (num_bits - 1)) - 1


def quantize_activation_per_token(x: Tensor, num_bits: int) -> tuple[Tensor, Tensor]:
    x32 = x.float()
    qmax = qmax_for_bits(num_bits)
    scale = qmax / x32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5)
    q = torch.round(x32 * scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_activation_per_token(q: Tensor, scale: Tensor, *, dtype: torch.dtype) -> Tensor:
    return (q.float() / scale.float().clamp_min(1e-5)).to(dtype=dtype)


def fake_quantize_activation_ste(x: Tensor, num_bits: int) -> Tensor:
    q, scale = quantize_activation_per_token(x, num_bits)
    dequantized = dequantize_activation_per_token(q, scale, dtype=x.dtype)
    return x + (dequantized - x).detach()


def quantize_ternary_tensor(t: Tensor, *, scale_mode: str) -> tuple[Tensor, Tensor]:
    _validate_scale_mode(scale_mode)
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError(f"ternary quantization expects a 2D tensor, got shape {tuple(t32.shape)}")
    if scale_mode == "per_row_absmean":
        scale = t32.abs().mean(dim=1).clamp_min(1e-5)
        q = torch.round(t32 / scale[:, None]).clamp(-1, 1).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    scale = torch.tensor(float(t32.abs().mean().clamp_min(1e-5).item()), dtype=torch.float32)
    q = torch.round(t32 / scale).clamp(-1, 1).to(torch.int8).contiguous()
    return q, scale


def dequantize_ternary_tensor(q: Tensor, scale: Tensor, *, dtype: torch.dtype, scale_mode: str) -> Tensor:
    _validate_scale_mode(scale_mode)
    if scale_mode == "per_row_absmean":
        return (q.float() * scale.float()[:, None]).to(dtype=dtype).contiguous()
    return (q.float() * float(scale.item())).to(dtype=dtype).contiguous()


def fake_ternary_weight_ste(weight: Tensor, *, scale_mode: str) -> Tensor:
    q, scale = quantize_ternary_tensor(weight, scale_mode=scale_mode)
    dequantized = dequantize_ternary_tensor(q, scale, dtype=weight.dtype, scale_mode=scale_mode)
    return weight + (dequantized - weight).detach()


def pack_ternary_codes(codes: Tensor) -> Tensor:
    if codes.dtype != torch.int8:
        raise ValueError(f"codes must have dtype torch.int8, got {codes.dtype}")
    flat = codes.reshape(-1).to(dtype=torch.int16)
    encoded = torch.full((flat.numel(),), 3, dtype=torch.uint8, device=flat.device)
    encoded = torch.where(flat == -1, torch.zeros_like(encoded), encoded)
    encoded = torch.where(flat == 0, torch.ones_like(encoded), encoded)
    encoded = torch.where(flat == 1, torch.full_like(encoded, 2), encoded)
    pad = (-encoded.numel()) % 4
    if pad:
        encoded = torch.cat(
            (encoded, torch.full((pad,), 3, dtype=torch.uint8, device=encoded.device)),
            dim=0,
        )
    encoded = encoded.reshape(-1, 4).to(dtype=torch.int32)
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    )
    return packed.to(dtype=torch.uint8).contiguous()


def unpack_ternary_codes(packed: Tensor, *, shape: tuple[int, int]) -> Tensor:
    packed_u8 = packed.reshape(-1).to(dtype=torch.uint8)
    decoded = torch.stack(
        (
            packed_u8 & 0x03,
            (packed_u8 >> 2) & 0x03,
            (packed_u8 >> 4) & 0x03,
            (packed_u8 >> 6) & 0x03,
        ),
        dim=1,
    ).reshape(-1)[: math.prod(shape)]
    codes = torch.zeros_like(decoded, dtype=torch.int8)
    codes = torch.where(decoded == 0, torch.full_like(codes, -1), codes)
    codes = torch.where(decoded == 2, torch.ones_like(codes), codes)
    return codes.reshape(shape).contiguous()


def dequantize_packed_ternary(
    packed: Tensor,
    scale: Tensor,
    *,
    shape: tuple[int, int],
    scale_mode: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    codes = unpack_ternary_codes(packed.to(device=device if device is not None else packed.device), shape=shape)
    scale = scale.to(device=codes.device)
    return dequantize_ternary_tensor(codes, scale, dtype=dtype, scale_mode=scale_mode)


def hadamard_transform_last_dim(x: Tensor) -> Tensor:
    dim = x.size(-1)
    if dim == 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard transform requires a power-of-two last dimension, got {dim}")
    shape = x.shape
    y = x.reshape(-1, dim)
    h = 1
    while h < dim:
        y = y.reshape(-1, dim // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.cat((a + b, a - b), dim=-1)
        h *= 2
    return (y.reshape(shape) / math.sqrt(dim)).contiguous()


@dataclass
class LowBitRuntimeState:
    packed_weight: Tensor
    scale: Tensor
    shape: tuple[int, int]
    scale_mode: str
    backend: str
    activation_bits: int | None = None
    apply_rmsnorm: bool = False
    hadamard: bool = False
    cache: dict[tuple[str, str, str], Tensor] = field(default_factory=dict)


def preprocess_lowbit_input(
    x: Tensor,
    *,
    activation_bits: int | None,
    apply_rmsnorm: bool,
    hadamard: bool,
    rmsnorm_fn,
) -> Tensor:
    y = x
    if apply_rmsnorm:
        y = rmsnorm_fn(y, (y.size(-1),))
    if hadamard:
        y = hadamard_transform_last_dim(y)
    if activation_bits is not None:
        y = fake_quantize_activation_ste(y, activation_bits) if y.requires_grad else dequantize_activation_per_token(
            *quantize_activation_per_token(y, activation_bits),
            dtype=y.dtype,
        )
    return y


if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _triton_bf16_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_remaining = K
        while k_remaining > 0:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            k_remaining -= BLOCK_K
        c = acc.to(tl.bfloat16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_linear_bf16(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    if triton is None or not x.is_cuda:
        out = F.linear(x, weight, bias)
        return out
    original_shape = x.shape[:-1]
    x2d = x.reshape(-1, x.size(-1)).contiguous().to(dtype=torch.bfloat16)
    w = weight.contiguous().to(dtype=torch.bfloat16)
    b = w.transpose(0, 1).contiguous()
    M, K = x2d.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x2d.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _triton_bf16_matmul_kernel[grid](
        x2d,
        b,
        out,
        M,
        N,
        K,
        x2d.stride(0),
        x2d.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out.reshape(*original_shape, N).to(dtype=x.dtype)


def lowbit_linear_forward(
    x: Tensor,
    state: LowBitRuntimeState,
    *,
    bias: Tensor | None,
    rmsnorm_fn,
) -> Tensor:
    processed = preprocess_lowbit_input(
        x,
        activation_bits=state.activation_bits,
        apply_rmsnorm=state.apply_rmsnorm,
        hadamard=state.hadamard,
        rmsnorm_fn=rmsnorm_fn,
    )
    cache_key = (str(processed.device), str(processed.dtype), state.backend)
    weight = state.cache.get(cache_key)
    if weight is None:
        weight = dequantize_packed_ternary(
            state.packed_weight,
            state.scale,
            shape=state.shape,
            scale_mode=state.scale_mode,
            dtype=processed.dtype,
            device=processed.device,
        )
        state.cache[cache_key] = weight
    if state.backend == "reference":
        return F.linear(processed, weight, bias.to(dtype=processed.dtype) if bias is not None else None)
    if state.backend == "triton_h100":
        return _triton_linear_bf16(processed, weight, bias)
    raise ValueError(f"Unsupported low-bit backend: {state.backend!r}")
