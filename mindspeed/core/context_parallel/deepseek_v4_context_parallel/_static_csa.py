# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

"""Host-owned fixed sequence plans and device-side CSA index sanitization."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeepSeekV4CPStaticCSAPlan:
    """Describe equal-length BSND batches with one contiguous local query range.

    The attention input producer creates this plan from its host-known shapes
    and partition offsets. It must not be inferred from arbitrary device
    metadata or reused after that metadata changes. The plan stores no tensors.
    """

    batch_size: int
    query_length: int
    global_length: int
    query_start: int
    compression_ratio: int = 4

    def __post_init__(self):
        for name in (
            "batch_size",
            "query_length",
            "global_length",
            "query_start",
            "compression_ratio",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be a Python int.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.query_length < 0 or self.global_length < 0 or self.query_start < 0:
            raise ValueError("query_length, global_length, and query_start must be non-negative.")
        if self.query_end > self.global_length:
            raise ValueError("The local query range must stay inside the global sequence.")
        if self.compression_ratio != 4:
            raise ValueError("The static CSA plan requires compression_ratio=4.")

    @property
    def query_end(self):
        return self.query_start + self.query_length

    @property
    def prefix_blocks(self):
        return self.query_end // self.compression_ratio

    @property
    def global_blocks(self):
        return self.global_length // self.compression_ratio

    @property
    def residual(self):
        return self.query_end % self.compression_ratio

    @property
    def global_cu_values(self):
        return tuple(index * self.global_length for index in range(self.batch_size + 1))


def sanitize_static_csa_indices(indices, plan, fill_empty_rows=False):
    """Mask invalid or future compressed keys without changing ``[B, S, K]``.

    Input indices are sample-relative compressed-key ordinals. Invalid entries
    become ``-1`` and output uses int32. For SMLA callers requesting the legacy
    empty-row fallback, a row with visible keys but no selected key receives
    key zero in its first slot. Rows with no visible key remain entirely -1.
    No values or dynamic lengths are read back from the device.
    """
    if not isinstance(plan, DeepSeekV4CPStaticCSAPlan):
        raise TypeError("plan must be a DeepSeekV4CPStaticCSAPlan.")
    if not torch.is_tensor(indices):
        raise TypeError("indices must be a torch.Tensor.")
    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, query, topk].")
    if indices.shape[:2] != (plan.batch_size, plan.query_length):
        raise ValueError("indices batch and query dimensions must match the static CSA plan.")
    if indices.shape[-1] == 0:
        raise ValueError("indices must contain at least one Top-K slot.")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("indices must use an integer dtype.")
    if type(fill_empty_rows) is not bool:
        raise TypeError("fill_empty_rows must be a Python bool.")

    positions = torch.arange(plan.query_length, device=indices.device, dtype=torch.int64)
    visible = torch.div(
        positions + plan.query_start + 1,
        plan.compression_ratio,
        rounding_mode="floor",
    ).view(1, plan.query_length, 1)
    # CANN already returns int32: do not allocate a full int64 Top-K buffer.
    # Promote narrow integer inputs for comparisons and retain int64 inputs
    # until after masking, so arbitrary large invalid ids cannot wrap around.
    ordinal_indices = indices if indices.dtype in (torch.int32, torch.int64) else indices.to(dtype=torch.int32)
    valid = (ordinal_indices >= 0) & (ordinal_indices < visible) & (ordinal_indices < plan.global_blocks)
    # Mask before narrowing so invalid large int64 inputs cannot wrap into valid ids.
    sanitized = torch.where(valid, ordinal_indices, -1).to(dtype=torch.int32)
    if fill_empty_rows:
        empty_visible = ~valid.any(dim=-1, keepdim=True) & (visible > 0)
        first_column = torch.arange(indices.shape[-1], device=indices.device) == 0
        sanitized = torch.where(empty_visible & first_column.view(1, 1, -1), 0, sanitized)
    return sanitized
