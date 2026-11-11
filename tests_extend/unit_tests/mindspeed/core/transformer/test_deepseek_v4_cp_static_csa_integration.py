# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Real-tensor integration tests without importing Megatron/torch_npu.

Load production function ASTs to isolate package initialization only. All tensor
operations and autograd are real PyTorch; the Indexer oracle replaces the CANN
kernel, not the production layout/metadata code. Run with unittest discovery.
"""

import argparse
import ast
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional, Protocol, Sequence
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[5]
CP = ROOT / "mindspeed/core/context_parallel/deepseek_v4_context_parallel"
ATTENTION = ROOT / "mindspeed/core/transformer/deepseek_v4/attention.py"
_SPEC = importlib.util.spec_from_file_location("_static_csa_integration_target", CP / "_static_csa.py")
_HELPER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPER
_SPEC.loader.exec_module(_HELPER)
Plan = _HELPER.DeepSeekV4CPStaticCSAPlan


def load_definitions(path, namespace, names=None, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    definitions = [
        node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (names is None or node.name in names)
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)


def production_namespace():
    ns = dict(torch=torch, dataclass=dataclass, lru_cache=lru_cache, Optional=Optional,
              Sequence=Sequence, Protocol=Protocol, __name__=__name__,
              DeepSeekV4CPStaticCSAPlan=Plan,
              sanitize_static_csa_indices=_HELPER.sanitize_static_csa_indices)
    for name in ("_types.py", "_utils.py", "_validation.py", "_smla_inputs.py"):
        load_definitions(CP / name, ns)
    load_definitions(ROOT / "mindspeed/core/transformer/deepseek_v4/cp_indexer.py", ns)
    load_definitions(ATTENTION, ns, {"_build_runtime_sequence_metadata"})
    load_definitions(CP / "ops/_stride_utils.py", ns)
    load_definitions(CP / "ops/npu_lightning_indexer.py", ns)
    return ns


class DenseIndexer:
    """Independent scoring oracle with the CANN right-down prefix convention."""

    def __init__(self):
        self.calls = []

    def forward_with_scores_compress(self, q, k, weights, packed, topk, offset, ratio,
                                    cmp_residual_k=None, return_scores=False, cmp_residual_k_values=None):
        residuals = cmp_residual_k_values if cmp_residual_k_values is not None else cmp_residual_k.tolist()
        self.calls.append((k.shape[0], tuple(residuals)))
        scores = torch.einsum("sbhd,kbd->bshk", q.double(), k[:, :, 0].double()).relu()
        scores = (scores * weights.permute(1, 0, 2).double().unsqueeze(-1)).sum(2)
        for batch, residual in enumerate(residuals):
            last_position = k.shape[0] * ratio + residual - 1
            for row in range(q.shape[0]):
                position = last_position - q.shape[0] + 1 + row
                for block in range(k.shape[0]):
                    if (block + 1) * ratio > position + 1:
                        scores[batch, row, block] = -float("inf")
        values, indices = scores.topk(min(topk, k.shape[0]), dim=-1)
        indices = torch.where(values.isfinite(), indices + offset, -1).int()
        indices = torch.nn.functional.pad(indices, (0, topk - indices.shape[-1]), value=-1)
        return indices, None


class TestStaticCSAIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = production_namespace()

    def test_runtime_producer_includes_tp_sequence_offset(self):
        build = self.ns["_build_runtime_sequence_metadata"]
        for batch in (1, 3):
            for cp_rank in range(8):
                for tp_rank in range(2):
                    for restored_query in (False, True):
                        query_length = 8 if restored_query else 4
                        q = torch.empty(batch, query_length, 2, 3)
                        kv = torch.empty(batch, 8, 1, 3)
                        result = build(q, kv, query_length, batch, 8, cp_rank, 2, tp_rank, True,
                                       torch.device("cpu"), return_static_csa_plan=True)
                        cu_q, cu_kv, cu_global, positions, kv_offset, plan = result
                        expected_start = cp_rank * 8 + (0 if restored_query else tp_rank * 4)
                        self.assertEqual(plan.query_start, expected_start)
                        self.assertEqual(kv_offset, cp_rank * 8)
                        self.assertEqual(cu_global.tolist(), list(plan.global_cu_values))
                        expected = [sample * 64 + expected_start + row
                                    for sample in range(batch) for row in range(query_length)]
                        self.assertEqual(positions.tolist(), expected)
                        self.assertEqual(cu_q[-1], batch * query_length)
                        self.assertEqual(cu_kv[-1], batch * 8)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            build(torch.empty(1, 4, 2, 3), torch.empty(1, 7, 1, 3), 4, 1, 8, 0, 2, 0,
                  True, torch.device("cpu"), return_static_csa_plan=True)

    def test_static_indexer_matches_dynamic_path_and_right_aligned_oracle(self):
        run = self.ns["run_deepseek_v4_right_aligned_fused_indexer"]
        torch.manual_seed(8321)
        for batch in (1, 2):
            for start, length in ((0, 2), (0, 9), (5, 8), (8, 8), (9, 5), (13, 6), (24, 8)):
                plan = Plan(batch, length, 32, start)
                q = torch.randn(length, batch, 3, 4)
                weights = torch.randn(length, batch, 3)
                keys = torch.randn(batch, 8, 1, 4)
                hidden = torch.zeros(length, batch, 5)
                cu = torch.tensor(plan.global_cu_values, dtype=torch.int32)
                positions = torch.arange(start, start + length)
                baseline, optimized = DenseIndexer(), DenseIndexer()
                args = (hidden, q, keys, weights, positions, cu, 4, 12, 0)
                old = run(baseline, *args, layout="BSND")
                new = run(optimized, *args, layout="BSND", static_csa_plan=plan)
                torch.testing.assert_close(new, old, rtol=0, atol=0)
                self.assertEqual(optimized.calls, baseline.calls)
                self.assertEqual(tuple(new.shape), (batch, length, 12))

    def make_smla(self, plan, full, indices, static, identity=True, shared=True, metadata=None):
        q = torch.randn(plan.batch_size, plan.query_length, 2, full.shape[-1])
        ori = torch.randn(plan.batch_size, plan.query_length + 4, 1, full.shape[-1])
        cu_q = torch.arange(plan.batch_size + 1, dtype=torch.int32) * plan.query_length
        cu_ori = torch.arange(plan.batch_size + 1, dtype=torch.int32) * ori.shape[1]
        positions = (torch.arange(plan.query_start, plan.query_end).unsqueeze(0)
                     + torch.arange(plan.batch_size).unsqueeze(1) * plan.global_length).reshape(-1)
        runtime = self.ns["build_deepseek_v4_owned_runtime_metadata"](
            q, "BSND", cu_q, cu_ori, torch.tensor(plan.global_cu_values, dtype=torch.int32),
            positions, 0, plan.batch_size, 4, static_csa_plan=plan if static else None,
        )
        prepared = SimpleNamespace(metadata=SimpleNamespace(is_identity_compact_order=identity,
                                                           batch_shared_sequence=shared))
        result = self.ns["build_deepseek_v4_cp_smla_inputs"](
            q, ori, 4, runtime, prepared, indices, True, "BSND", "BSND",
            metadata=metadata,
            compacted_compressed_kv=full, compacted_block_starts=torch.arange(plan.global_blocks) * 4,
        )
        return result

    def test_non_identity_or_non_shared_metadata_uses_general_path(self):
        plan = Plan(2, 7, 32, 8)
        full = torch.randn(2, 8, 1, 3)
        # The generic validation path requires a CANN-supported Top-K width.
        indices = torch.zeros(2, 7, 512, dtype=torch.int32)
        for identity, shared in ((False, True), (True, False), (False, False)):
            with mock.patch.dict(self.ns, {"_build_static_csa_smla_inputs": mock.Mock(
                side_effect=AssertionError("unexpected static dispatch")
            )}):
                result = self.make_smla(plan, full, indices, static=True, identity=identity, shared=shared)
            torch.testing.assert_close(result.cmp_kv, full[:, :plan.prefix_blocks])

    def test_changed_plan_rebuilds_prefix_and_rejects_stale_metadata(self):
        full = torch.randn(1, 8, 1, 3)
        indices = torch.zeros(1, 7, 5, dtype=torch.int32)
        first, second = Plan(1, 7, 32, 8), Plan(1, 7, 32, 13)
        a = self.make_smla(first, full, indices, static=True)
        b = self.make_smla(second, full, indices, static=True)
        self.assertNotEqual(a.cmp_kv.shape[1], b.cmp_kv.shape[1])
        self.assertEqual(a.cmp_residual_kv.tolist(), [3])
        self.assertEqual(b.cmp_residual_kv.tolist(), [0])
        with self.assertRaisesRegex(ValueError, "metadata=None"):
            self.make_smla(first, full, indices, static=True, metadata=torch.zeros(1))

    def test_smla_prefix_indices_metadata_and_gradient_parity(self):
        for batch in (1, 3):
            for start in (0, 5, 8, 17):
                plan = Plan(batch, 7, 32, start)
                full_old = torch.randn(batch, 8, 1, 3, dtype=torch.double, requires_grad=True)
                full_new = full_old.detach().clone().requires_grad_(True)
                # Kernel-valid selections plus empty rows exercise the fallback.
                indices = torch.full((batch, 7, 5), -1, dtype=torch.int32)
                for row in range(7):
                    if (start + row + 1) // 4 and row % 2:
                        indices[:, row, 0] = (start + row + 1) // 4 - 1
                old = self.make_smla(plan, full_old, indices, static=False)
                new = self.make_smla(plan, full_new, indices, static=True)
                for field in ("cmp_kv", "cmp_sparse_indices", "cu_seqlens_cmp_kv", "cmp_residual_kv", "block_starts"):
                    torch.testing.assert_close(getattr(new, field), getattr(old, field), rtol=0, atol=0)
                for field in ("global_to_local", "selected_global_indices"):
                    torch.testing.assert_close(getattr(new.alignment, field), getattr(old.alignment, field))
                old.cmp_kv.square().sum().backward()
                new.cmp_kv.square().sum().backward()
                torch.testing.assert_close(full_new.grad, full_old.grad, rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(full_new.grad[:, plan.prefix_blocks:]), 0)
                # Indexer loss must crop its keys through the identical descriptor.
                indexer_keys = torch.randn(batch, 8, 1, 5, requires_grad=True)
                loss_keys = self.ns["align_deepseek_v4_cp_tensor"](indexer_keys, new.alignment, 1)
                loss_keys.sum().backward()
                self.assertTrue(torch.equal(indexer_keys.grad[:, :plan.prefix_blocks],
                                            torch.ones_like(indexer_keys.grad[:, :plan.prefix_blocks])))
                self.assertEqual(torch.count_nonzero(indexer_keys.grad[:, plan.prefix_blocks:]), 0)

    def test_operator_wrapper_host_residual_matches_legacy_and_avoids_readback(self):
        ns = production_namespace()
        calls = []

        def metadata_op(*args, **kwargs):
            calls.append(("metadata", kwargs))
            return torch.zeros(1, dtype=torch.int32)

        def indexer_op(q, k, w, topk, **kwargs):
            calls.append(("indexer", kwargs))
            return torch.zeros(q.shape[0], q.shape[1], topk, dtype=torch.int32), None

        ns["_CUSTOM_OPS"] = SimpleNamespace(lightning_indexer_metadata=metadata_op, lightning_indexer=indexer_op)
        run = ns["npu_lightning_indexer"]
        q, k, w = torch.randn(7, 2, 3, 4), torch.randn(5, 2, 1, 4), torch.randn(7, 2, 3)
        run(q, k, w, 8, cmp_residual_k=torch.tensor([3, 3], dtype=torch.int32))
        expected_residual = calls[-1][1]["cmp_residual_k"].clone()
        calls.clear()
        with ExitStack() as stack:
            for name in ("item", "cpu", "tolist", "__bool__"):
                stack.enter_context(mock.patch.object(torch.Tensor, name, side_effect=AssertionError(name)))
            stack.enter_context(mock.patch.object(torch, "nonzero", side_effect=AssertionError("nonzero")))
            stack.enter_context(mock.patch.object(torch, "equal", side_effect=AssertionError("equal")))
            run(q, k, w, 8, cmp_residual_k_values=(3, 3))
        self.assertEqual([name for name, _ in calls], ["indexer"])
        torch.testing.assert_close(calls[-1][1]["cmp_residual_k"], expected_residual)
        with self.assertRaisesRegex(ValueError, "either"):
            run(q, k, w, 8, cmp_residual_k=torch.tensor([3, 3]), cmp_residual_k_values=(3, 3))
        with self.assertRaisesRegex(ValueError, "per sample"):
            run(q, k, w, 8, cmp_residual_k_values=(4, 4))

    def test_static_smla_has_no_host_value_reads(self):
        plan = Plan(2, 7, 32, 9)
        full = torch.randn(2, 8, 1, 4)
        indices = torch.full((2, 7, 6), -1, dtype=torch.int32)
        with ExitStack() as stack:
            for name in ("item", "cpu", "tolist", "__bool__"):
                stack.enter_context(mock.patch.object(torch.Tensor, name, side_effect=AssertionError(name)))
            stack.enter_context(mock.patch.object(torch, "nonzero", side_effect=AssertionError("nonzero")))
            stack.enter_context(mock.patch.object(torch, "equal", side_effect=AssertionError("equal")))
            new = self.make_smla(plan, full, indices, static=True)
        self.assertEqual(tuple(new.cmp_sparse_indices.shape), (2, 7, 1, 6))

    def test_flag_is_opt_in_and_validated_in_both_entrypoints(self):
        for relative in (
            "mindspeed/arguments.py",
            "mindspeed/features_manager/context_parallel/context_parallel_feature.py",
        ):
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     and node.args and isinstance(node.args[0], ast.Constant)
                     and node.args[0].value == "--deepseek-v4-cp-static-csa"]
            self.assertEqual(len(calls), 1)
            parser = argparse.ArgumentParser()
            parser.add_argument(*(ast.literal_eval(arg) for arg in calls[0].args),
                                **{kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords})
            self.assertFalse(parser.parse_args([]).deepseek_v4_cp_static_csa)
            self.assertTrue(parser.parse_args(["--deepseek-v4-cp-static-csa"]).deepseek_v4_cp_static_csa)
            guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name)
                      and node.test.func.id == "getattr" and len(node.test.args) > 1
                      and isinstance(node.test.args[1], ast.Constant)
                      and node.test.args[1].value == "deepseek_v4_cp_static_csa"]
            self.assertEqual(len(guards), 1)
            code = compile(ast.Module(body=guards, type_ignores=[]), relative, "exec")
            for enabled, algo, size, fused, raises in (
                (True, "deepseek_v4_cp_algo", 8, True, False),
                (True, "deepseek_v4_cp_algo", 1, True, True),
                (True, "ulysses_cp_algo", 8, True, True),
                (True, "deepseek_v4_cp_algo", 8, False, True),
                (False, "ulysses_cp_algo", 1, False, False),
            ):
                args = SimpleNamespace(deepseek_v4_cp_static_csa=enabled, context_parallel_algo=algo,
                                       context_parallel_size=size, use_fused_lightning_indexer=fused)
                if raises:
                    with self.assertRaises(AssertionError):
                        exec(code, {"args": args})
                else:
                    exec(code, {"args": args})


if __name__ == "__main__":
    unittest.main()
