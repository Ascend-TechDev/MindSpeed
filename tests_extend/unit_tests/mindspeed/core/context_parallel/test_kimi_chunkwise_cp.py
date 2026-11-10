# Copyright (c) 2026, the MindSpeed contributors.
# SPDX-License-Identifier: Apache-2.0

"""Run with torchrun on two NPUs to validate the model-side CP example."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

pytest.importorskip('torch_npu')
pytest.importorskip('fla')

from examples.kimi_chunkwise_cp.attention import _context, kda_attention_chunkwise
from fla.utils import assert_close


@pytest.fixture(scope='module', name='cp_groups')
def initialize_cp_groups():
    if int(os.environ.get('WORLD_SIZE', '1')) != 2 or not torch.npu.is_available():
        pytest.skip('Run with torchrun --nproc-per-node=2 on Ascend NPUs')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(minutes=5))
    singleton = None
    for rank in range(2):
        group = dist.new_group([rank], backend='hccl')
        if dist.get_rank() == rank:
            singleton = group
    yield dist.group.WORLD, singleton
    _context.cache_clear()
    dist.destroy_process_group(singleton)
    dist.destroy_process_group()


def make_attention():
    torch.manual_seed(42)
    hidden, heads, dim, width = 64, 2, 128, 4
    attention = nn.Module()

    def linear(input_size, output_size):
        return nn.Linear(input_size, output_size, bias=False, dtype=torch.bfloat16)

    for name in ('q', 'k', 'v'):
        attention.add_module(name + '_proj', linear(hidden, heads * dim))
        attention.add_module(
            name + '_conv1d',
            nn.Conv1d(heads * dim, heads * dim, width, groups=heads * dim, bias=False),
        )
    attention.f_a_proj = linear(hidden, 16)
    attention.f_b_proj = linear(16, heads * dim)
    attention.b_proj = linear(hidden, heads)
    attention.g_proj = linear(hidden, heads * dim)
    attention.o_proj = linear(heads * dim, hidden)
    attention.A_log = nn.Parameter(torch.zeros(heads))
    attention.dt_bias = nn.Parameter(torch.zeros(heads * dim))
    attention.o_norm = nn.Module()
    attention.o_norm.weight = nn.Parameter(torch.ones(dim))
    config = {
        'rms_norm_eps': 1e-5,
        'linear_attn_config': {
            'num_heads': heads,
            'head_dim': dim,
            'short_conv_kernel_size': width,
            'gate_lower_bound': -5,
        },
    }
    return attention.to('npu'), config


@pytest.mark.parametrize('recompute', [False, True], ids=['eager', 'reentrant_checkpoint'])
def test_chunkwise_outputs_and_gradients(cp_groups, recompute):
    group, singleton = cp_groups
    attention, config = make_attention()
    torch.manual_seed(123)
    x = torch.randn(1, 512, 64, dtype=torch.bfloat16).to('npu').requires_grad_()
    dy = torch.randn_like(x.cpu()).to('npu')
    parameters = tuple(attention.parameters())
    reference = kda_attention_chunkwise(x, attention, config, singleton)
    reference.backward(dy)
    reference_dx = x.grad.detach().clone()
    reference_grads = [parameter.grad.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = None

    start = dist.get_rank() * 256
    local = x.detach()[:, start : start + 256].clone().requires_grad_()
    calls = []

    def forward(value):
        calls.append(True)
        return kda_attention_chunkwise(value, attention, config, group)

    output = checkpoint(forward, local, use_reentrant=True) if recompute else forward(local)
    output.backward(dy[:, start : start + 256].contiguous())
    assert len(calls) == (2 if recompute else 1)
    assert torch.isfinite(output).all()
    assert torch.isfinite(local.grad).all()
    assert_close('output', reference[:, start : start + 256], output, 0.005)
    assert_close('input gradient', reference_dx[:, start : start + 256], local.grad, 0.025)
    for (name, parameter), expected in zip(attention.named_parameters(), reference_grads):
        gradient = parameter.grad.float()
        dist.all_reduce(gradient, group=group)
        assert torch.isfinite(gradient).all(), name
        assert_close(name, expected.float(), gradient, 0.025)


@pytest.mark.parametrize('shape', [(2, 16, 64), (1, 0, 64), (16, 64)])
def test_invalid_continuous_input(shape):
    with pytest.raises(ValueError, match='one nonempty continuous sequence'):
        kda_attention_chunkwise(torch.empty(shape), None, {}, None)


def test_explicit_cp_group_required():
    with pytest.raises(ValueError, match='context-parallel group explicitly'):
        kda_attention_chunkwise(torch.empty(1, 16, 64), None, {}, None)
