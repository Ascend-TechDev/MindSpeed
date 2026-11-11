# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Single-NPU parity checks for the real CSA Query/weights stream helpers.

Run this file directly with the training environment's Python. The tests load
the production methods and rotary/Hadamard functions from their AST to avoid
requiring a Megatron model or process group. Only the two projection modules
are replaced by small nn.Linear modules. This does not test HCCL ordering or
replace a CP8/TP2 training, gradient, and performance comparison.
"""

import ast
from functools import lru_cache
import math
from pathlib import Path
import unittest


IMPORT_ERROR = None
try:
    import torch
    import torch_npu  # noqa: F401
    from einops import rearrange
    from scipy.linalg import hadamard
    from torch.utils.checkpoint import checkpoint
except ImportError as error:
    IMPORT_ERROR = str(error)


SOURCE_DIR = (
    Path(__file__).resolve().parents[5]
    / "mindspeed/core/transformer/deepseek_v4"
)


def load_functions(filename, names, namespace, class_name=None):
    tree = ast.parse((SOURCE_DIR / filename).read_text(encoding="utf-8"))
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Production methods not found in {filename}: {names}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_DIR / filename), "exec"), namespace)


@unittest.skipIf(IMPORT_ERROR is not None, f"NPU test dependencies unavailable: {IMPORT_ERROR}")
class TestDeepSeekV4CPQueryStreamNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise unittest.SkipTest("A working Ascend NPU is required")
        cls.device = torch.device("npu", torch.npu.current_device())
        namespace = {
            "torch": torch, "math": math, "lru_cache": lru_cache,
            "hadamard": hadamard, "rearrange": rearrange,
        }
        load_functions(
            "deepseek_utils.py",
            {"apply_rotary_emb", "rotate_activation", "hadamard_transform_ref", "get_hadamard_tensor"},
            namespace,
        )
        load_functions("indexer.py", {"forward_query_weights"}, namespace, "DSAIndexer")
        helper_names = {"_launch_indexer_query_weights", "_wait_indexer_query_weights"}
        load_functions("attention.py", helper_names, namespace, "DeepSeekV4SelfAttentionCP")
        cls.Harness = type("QueryStreamHarness", (), {name: namespace[name] for name in helper_names})
        real_forward = namespace["forward_query_weights"]

        class QueryIndexer(torch.nn.Module):
            forward_query_weights = real_forward

            def __init__(self):
                super().__init__()
                self.n_heads = 4
                self.head_dim = 128
                self.rope_head_dim = 64
                self.softmax_scale = self.head_dim ** -0.5
                self.wq_b = torch.nn.Linear(32, self.n_heads * self.head_dim, bias=False)
                self.weights_proj = torch.nn.Linear(64, self.n_heads, bias=False)
                self.observed_grad_modes = []
                self.wq_b.register_forward_pre_hook(self._observe_grad_mode)

            def _observe_grad_mode(self, _module, _inputs):
                self.observed_grad_modes.append(torch.is_grad_enabled())

        cls.QueryIndexer = QueryIndexer

    def _run_microbatch(self, subject, inputs, *, overlap, recompute, consumer):
        hidden, query_lora, freqs, main_input = inputs
        subject.indexer.zero_grad(set_to_none=True)
        producer = torch.npu.current_stream(self.device)
        consumer.wait_stream(producer)

        def forward(hidden_arg, query_arg, main_arg):
            if overlap:
                pending = subject._launch_indexer_query_weights(hidden_arg, query_arg, freqs)
                subject.events.append(pending[2])
            else:
                query, weights = subject.indexer.forward_query_weights(
                    hidden_arg.detach(), query_arg.detach(), freqs,
                )
            # Independent work is submitted on the caller's stream before its
            # join with the Query stream. This branch keeps reentrant checkpoint
            # inputs connected while the Indexer inputs must remain detached.
            main_output = main_arg.square() + 0.125 * main_arg
            if overlap:
                query, weights = subject._wait_indexer_query_weights(pending)
            return main_output, query, weights

        with torch.npu.stream(consumer):
            if recompute:
                outputs = checkpoint(forward, hidden, query_lora, main_input, use_reentrant=True)
            else:
                outputs = forward(hidden, query_lora, main_input)
            main_output, query, weights = outputs
            loss = main_output.float().square().mean()
            loss = loss + query.float().square().mean() + 0.7 * weights.float().square().mean()
            loss.backward()

        # Synchronization is only for inspecting the test result; production
        # helpers must establish their own event and allocator dependencies.
        torch.npu.synchronize(self.device)
        self.assertIsNone(hidden.grad, "Indexer must not backpropagate into hidden inputs")
        self.assertIsNone(query_lora.grad, "Indexer must not backpropagate into Q low-rank inputs")
        self.assertIsNotNone(main_input.grad, "Checkpoint's independent main branch must participate in backward")
        gradients = {}
        for name, parameter in subject.indexer.named_parameters():
            self.assertIsNotNone(parameter.grad, f"Missing Indexer parameter gradient: {name}")
            gradients[name] = parameter.grad.detach().float().cpu()
            self.assertTrue(torch.isfinite(gradients[name]).all().item())
            self.assertGreater(gradients[name].abs().sum().item(), 0.0)
        return {
            "query": query.detach().float().cpu(),
            "weights": weights.detach().float().cpu(),
            "main": main_output.detach().float().cpu(),
            "main_grad": main_input.grad.detach().float().cpu(),
            **gradients,
        }

    def _check_parity(self, dtype, *, recompute, non_default_consumer):
        torch.manual_seed(1234)
        baseline = self.Harness()
        baseline.indexer = self.QueryIndexer().to(device=self.device, dtype=dtype)
        baseline._indexer_query_weights_stream = None
        baseline.events = []
        side = self.Harness()
        side.indexer = self.QueryIndexer().to(device=self.device, dtype=dtype)
        side.indexer.load_state_dict(baseline.indexer.state_dict())
        side._indexer_query_weights_stream = None
        side.events = []
        default = torch.npu.current_stream(self.device)
        consumer = torch.npu.Stream(device=self.device) if non_default_consumer else default
        cached_stream = None

        # Different sequence lengths exercise independent microbatches while
        # reusing the cached stream. Retain completion events to check identity.
        for seq_len in (16, 24):
            hidden_cpu = torch.randn(seq_len, 2, 64)
            query_cpu = torch.randn(seq_len, 2, 32)
            main_cpu = torch.randn(32, 32)
            phases = torch.arange(seq_len, dtype=torch.float32)[:, None]
            phases = phases * torch.linspace(0.01, 0.2, 32)[None, :]
            freqs = torch.polar(torch.ones_like(phases), phases).to(self.device)

            def make_inputs():
                return (
                    hidden_cpu.to(device=self.device, dtype=dtype).detach().requires_grad_(True),
                    query_cpu.to(device=self.device, dtype=dtype).detach().requires_grad_(True),
                    freqs,
                    main_cpu.to(device=self.device, dtype=dtype).detach().requires_grad_(True),
                )

            expected = self._run_microbatch(
                baseline, make_inputs(), overlap=False, recompute=recompute, consumer=consumer,
            )
            actual = self._run_microbatch(
                side, make_inputs(), overlap=True, recompute=recompute, consumer=consumer,
            )
            for name in expected:
                with self.subTest(tensor=name, seq_len=seq_len):
                    # Projection gradients are much smaller than activations;
                    # an activation-sized atol could hide a broken backward.
                    is_output = name in {"query", "weights", "main"}
                    if dtype == torch.bfloat16:
                        tolerances = {"atol": 2e-3 if is_output else 1e-5, "rtol": 2e-2}
                    else:
                        tolerances = {"atol": 1e-5 if is_output else 1e-6, "rtol": 1e-4}
                    torch.testing.assert_close(actual[name], expected[name], **tolerances)
            self.assertIsNone(baseline._indexer_query_weights_stream)
            self.assertIsNotNone(side._indexer_query_weights_stream)
            if cached_stream is not None:
                self.assertIs(side._indexer_query_weights_stream, cached_stream)
            cached_stream = side._indexer_query_weights_stream

        expected_modes = [False, True, False, True] if recompute else [True, True]
        self.assertEqual(baseline.indexer.observed_grad_modes, expected_modes)
        self.assertEqual(side.indexer.observed_grad_modes, expected_modes)
        self.assertEqual(len(side.events), len(expected_modes))
        self.assertEqual(len({id(event) for event in side.events}), len(side.events))

    def test_forward_and_parameter_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for non_default_consumer in (False, True):
                with self.subTest(dtype=dtype, non_default_consumer=non_default_consumer):
                    self._check_parity(dtype, recompute=False, non_default_consumer=non_default_consumer)

    def test_reentrant_checkpoint_grad_mode_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for non_default_consumer in (False, True):
                with self.subTest(dtype=dtype, non_default_consumer=non_default_consumer):
                    self._check_parity(dtype, recompute=True, non_default_consumer=non_default_consumer)


if __name__ == "__main__":
    unittest.main()
