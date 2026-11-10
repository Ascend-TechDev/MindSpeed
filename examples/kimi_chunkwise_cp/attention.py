# Copyright (c) 2026, the MindSpeed contributors.
# SPDX-License-Identifier: Apache-2.0

"""Kimi KDA attention with contiguous-token FLA context parallelism."""

from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F

from fla.modules.causal_conv1d import causal_conv1d
from fla.modules.norm.fused_norm_gate import rms_norm_gated
from fla.ops.cp import build_cp_context
from fla.ops.kda import chunk_kda


@lru_cache(maxsize=16)
def _context(tokens, width, device, group):
    total = tokens * dist.get_world_size(group)
    boundaries = torch.tensor([0, total], dtype=torch.long)
    return build_cp_context(boundaries.to(device), group=group, cu_seqlens_cpu=boundaries, conv1d_kernel_size=width)


def kda_attention_chunkwise(x, a, c, cp_group):
    """Use all heads on local tokens, exchanging convolution halos and KDA states."""
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] == 0:
        raise ValueError('Expected one nonempty continuous sequence with shape [1, local_tokens, hidden_size].')
    if not dist.is_initialized() or cp_group is None:
        raise ValueError('Pass the initialized framework context-parallel group explicitly.')
    b, t, _ = x.shape
    k = c['linear_attn_config']
    h, d = k['num_heads'], k['head_dim']
    context = _context(t, k['short_conv_kernel_size'], x.device, cp_group)
    qkv = []
    for name in ('q', 'k', 'v'):
        projected = F.linear(x, getattr(a, name + '_proj').weight)
        weight = getattr(a, name + '_conv1d').weight[:, 0, :].contiguous()
        convolved, _ = causal_conv1d(projected, weight, activation='silu', cp_context=context)
        qkv.append(convolved.view(b, t, h, d))
    raw_g = F.linear(F.linear(x, a.f_a_proj.weight), a.f_b_proj.weight).view(b, t, h, d)
    beta = F.linear(x, a.b_proj.weight).float()
    out, _ = chunk_kda(
        *qkv,
        raw_g,
        beta,
        A_log=a.A_log[:h],
        dt_bias=a.dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=k['gate_lower_bound'],
        output_final_state=False,
        cp_context=context,
    )
    gate = F.linear(x, a.g_proj.weight).view(b, t, h, d)
    out = rms_norm_gated(out.contiguous(), gate, a.o_norm.weight, None, activation='sigmoid', eps=c['rms_norm_eps'])
    return F.linear(out.flatten(2), a.o_proj.weight)
