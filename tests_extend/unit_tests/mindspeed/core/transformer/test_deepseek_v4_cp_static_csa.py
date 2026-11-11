# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU numerical tests for the static CSA layout contract using real PyTorch.

Load the helper directly so these tests do not import torch_npu or Megatron.
The Python oracle enumerates causally complete compression groups independently
of the tensor implementation. NPU integration and timing remain separate checks.
"""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import torch


HELPER_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "mindspeed/core/context_parallel/deepseek_v4_context_parallel/_static_csa.py"
)
_MODULE_NAME = "_mindspeed_static_csa_test_target"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, HELPER_SOURCE)
_HELPER = importlib.util.module_from_spec(_SPEC)
# dataclasses resolves annotations through the defining module.
sys.modules[_MODULE_NAME] = _HELPER
_SPEC.loader.exec_module(_HELPER)
Plan = _HELPER.DeepSeekV4CPStaticCSAPlan
sanitize = _HELPER.sanitize_static_csa_indices


def reference_indices(values, plan, fill_empty_rows=False):
    """A block is visible only when every source token precedes the query end."""
    result = []
    for sample in values:
        sample_result = []
        for offset, row in enumerate(sample):
            query_end = plan.query_start + offset + 1
            visible = {
                block
                for block in range(plan.global_length // plan.compression_ratio)
                if (block + 1) * plan.compression_ratio <= query_end
            }
            cleaned = [index if index in visible else -1 for index in row]
            if fill_empty_rows and visible and all(index == -1 for index in cleaned):
                cleaned[0] = 0
            sample_result.append(cleaned)
        result.append(sample_result)
    return result


def make_values(batch_size, query_length, width, global_blocks, seed):
    rng = random.Random(seed)
    values = []
    for batch in range(batch_size):
        rows = []
        for query in range(query_length):
            anchors = [-7, -1, 0, 1, global_blocks - 1, global_blocks, global_blocks + 9]
            rows.append(
                [anchors[(batch + query + col) % len(anchors)]
                 if col % 2 == 0 else rng.randint(-3, global_blocks + 3)
                 for col in range(width)]
            )
        values.append(rows)
    return values


class TestStaticCSAPlan(unittest.TestCase):
    def test_geometry_is_host_known_and_immutable(self):
        for start in range(8):
            for length in range(8):
                with self.subTest(start=start, length=length):
                    plan = Plan(3, length, 32, start)
                    self.assertEqual(plan.query_end, start + length)
                    self.assertEqual(plan.prefix_blocks, (start + length) // 4)
                    self.assertEqual(plan.global_blocks, 8)
                    self.assertEqual(plan.residual, (start + length) % 4)
                    self.assertEqual(tuple(plan.global_cu_values), (0, 32, 64, 96))
                    self.assertEqual(plan.compression_ratio, 4)
                    with self.assertRaises(FrozenInstanceError):
                        plan.query_start = 1

    def test_cp8_tp2_64k_host_geometry(self):
        global_length, cp_size, tp_size = 65536, 8, 2
        local_length = global_length // cp_size
        for cp_rank in range(cp_size):
            for tp_rank in range(tp_size):
                start = cp_rank * local_length + tp_rank * (local_length // tp_size)
                plan = Plan(2, local_length // tp_size, global_length, start)
                self.assertEqual(plan.query_end, start + 4096)
                self.assertEqual(plan.prefix_blocks, (start + 4096) // 4)
                self.assertEqual(plan.global_blocks, 16384)
                self.assertEqual(tuple(plan.global_cu_values), (0, 65536, 131072))

    def test_invalid_plan_parameters_are_rejected(self):
        base = dict(batch_size=1, query_length=8, global_length=16, query_start=0)
        invalid = [
            ("batch_size", 0), ("batch_size", -1), ("batch_size", True),
            ("batch_size", 1.0), ("query_length", -1), ("query_length", False),
            ("global_length", -1), ("global_length", 16.0),
            ("query_start", -1), ("query_start", True), ("query_start", 9),
            ("compression_ratio", 0), ("compression_ratio", 1),
            ("compression_ratio", 8), ("compression_ratio", 128),
            ("compression_ratio", 4.0), ("compression_ratio", True),
        ]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                args = dict(base, **{field: value})
                with self.assertRaises((TypeError, ValueError)):
                    Plan(**args)
        with self.assertRaises((TypeError, ValueError)):
            Plan(1, 1, 0, 0)
        with self.assertRaises((TypeError, ValueError)):
            Plan(torch.tensor(1), 8, 16, 0)


class TestStaticCSAIndices(unittest.TestCase):
    def assert_matches_oracle(self, values, plan, fill_empty_rows=False, dtype=torch.int64):
        indices = torch.tensor(values, dtype=dtype)
        original = indices.clone()
        actual = sanitize(indices, plan, fill_empty_rows=fill_empty_rows)
        expected = torch.tensor(reference_indices(values, plan, fill_empty_rows), dtype=torch.int32)
        self.assertEqual(actual.shape, indices.shape)
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(actual.device, indices.device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(indices, original, rtol=0, atol=0)
        return actual

    def test_oracle_for_batches_residuals_and_fixed_topk_widths(self):
        for batch_size in (1, 3):
            for start in (0, 1, 3, 8, 29):
                for residual in range(4):
                    length = 4 + ((residual - start) % 4)
                    for width in (1, 3, 11):
                        for fallback in (False, True):
                            with self.subTest(batch=batch_size, start=start, residual=residual,
                                              width=width, fallback=fallback):
                                plan = Plan(batch_size, length, 40, start)
                                values = make_values(batch_size, length, width, 10, seed=start + width)
                                self.assert_matches_oracle(values, plan, fallback)

    def test_causal_boundary_and_no_visible_block(self):
        plan = Plan(1, 8, 8, 0)
        values = [[[0, 1, -1, 7] for _ in range(8)]]
        actual = self.assert_matches_oracle(values, plan)
        torch.testing.assert_close(actual[0, :3], torch.full((3, 4), -1, dtype=torch.int32))
        self.assertEqual(actual[0, 3].tolist(), [0, -1, -1, -1])
        self.assertEqual(actual[0, 7].tolist(), [0, 1, -1, -1])
        short_plan = Plan(2, 3, 3, 0)
        short_values = [[[0, -1, 99] for _ in range(3)] for _ in range(2)]
        self.assert_matches_oracle(short_values, short_plan, fill_empty_rows=True)

    def test_fallback_only_fills_empty_visible_rows(self):
        plan = Plan(2, 3, 12, 3)
        values = [
            [[-1, 1, 99], [0, -1, 99], [-7, -1, 99]],
            [[0, -1, -1], [-1, -1, -1], [-1, 0, -1]],
        ]
        actual = self.assert_matches_oracle(values, plan, fill_empty_rows=True)
        self.assertEqual(actual[0, 0].tolist(), [0, -1, -1])
        self.assertEqual(actual[1, 2].tolist(), [-1, 0, -1])

    def test_int64_overflow_cannot_turn_invalid_indices_into_valid_blocks(self):
        plan = Plan(1, 1, 16, 15)
        values = [[[0, 3, 4, 2**32, 2**32 + 1, -(2**32), -(2**32) + 2, 2**63 - 1]]]
        self.assert_matches_oracle(values, plan)

    def test_supported_integer_dtypes_return_int32(self):
        plan = Plan(1, 2, 16, 6)
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                self.assert_matches_oracle([[[0, 1, 9], [1, 0, 4]]], plan, dtype=dtype)

    def test_uint8_indices_are_compared_against_full_width_bounds(self):
        # A device scalar bound of 256 must not narrow to uint8 zero.
        plan = Plan(1, 1, 2048, 1023)
        self.assert_matches_oracle([[[0, 127, 128, 255]]], plan, dtype=torch.uint8)
    def test_noncontiguous_input_preserves_shape_and_values(self):
        plan = Plan(2, 7, 32, 9)
        values = make_values(2, 7, 5, 8, seed=19)
        indices = torch.tensor(values, dtype=torch.int64).transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(indices.is_contiguous())
        actual = sanitize(indices, plan, fill_empty_rows=True)
        expected = torch.tensor(reference_indices(values, plan, True), dtype=torch.int32)
        self.assertEqual(actual.shape, (2, 7, 5))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cp8_tp_offsets_are_equivalent_to_unsplit_query(self):
        global_length, cp_size, local_length = 256, 8, 32
        for rank in range(cp_size):
            full_plan = Plan(2, local_length, global_length, rank * local_length)
            values = make_values(2, local_length, 7, global_length // 4, seed=rank)
            indices = torch.tensor(values, dtype=torch.int64)
            for fallback in (False, True):
                full = sanitize(indices, full_plan, fill_empty_rows=fallback)
                shards = []
                for offset in (0, 16):
                    shard_plan = Plan(2, 16, global_length, rank * local_length + offset)
                    shards.append(sanitize(indices[:, offset:offset + 16], shard_plan,
                                           fill_empty_rows=fallback))
                torch.testing.assert_close(torch.cat(shards, dim=1), full, rtol=0, atol=0)
                self.assert_matches_oracle(values, full_plan, fallback)

    def test_batch_samples_do_not_share_indices(self):
        plan = Plan(3, 2, 20, 15)
        values = [[[0, -1], [1, -1]], [[3, 2], [-1, 3]], [[99, -1], [-1, -1]]]
        together = self.assert_matches_oracle(values, plan, fill_empty_rows=True)
        for batch, sample in enumerate(values):
            one = sanitize(torch.tensor([sample]), Plan(1, 2, 20, 15), fill_empty_rows=True)
            torch.testing.assert_close(together[batch], one[0], rtol=0, atol=0)

    def test_empty_query_has_fixed_empty_shape(self):
        for global_length, start in ((0, 0), (16, 0), (16, 16)):
            plan = Plan(2, 0, global_length, start)
            for fallback in (False, True):
                indices = torch.empty((2, 0, 5), dtype=torch.int64)
                actual = sanitize(indices, plan, fill_empty_rows=fallback)
                self.assertEqual(actual.shape, (2, 0, 5))
                self.assertEqual(actual.dtype, torch.int32)

    def test_invalid_index_contract_is_rejected(self):
        plan = Plan(2, 3, 16, 0)
        bad_inputs = [
            [[[0]]], torch.zeros((2, 3), dtype=torch.int32),
            torch.zeros((2, 3, 1, 1), dtype=torch.int32),
            torch.zeros((1, 3, 4), dtype=torch.int32),
            torch.zeros((2, 2, 4), dtype=torch.int32),
            torch.zeros((2, 3, 0), dtype=torch.int32),
            torch.zeros((2, 3, 4), dtype=torch.float32),
            torch.zeros((2, 3, 4), dtype=torch.bool),
        ]
        for indices in bad_inputs:
            with self.subTest(shape=getattr(indices, "shape", None), dtype=getattr(indices, "dtype", None)):
                with self.assertRaises((TypeError, ValueError)):
                    sanitize(indices, plan)
        for fallback in (0, 1, None, "yes"):
            with self.subTest(fallback=fallback):
                with self.assertRaises((TypeError, ValueError)):
                    sanitize(torch.zeros((2, 3, 4), dtype=torch.int32), plan, fill_empty_rows=fallback)

    def test_meta_tensor_needs_shapes_only(self):
        # Meta tensors have no readable values; fixed output geometry must work.
        for length in (0, 5):
            plan = Plan(2, length, 32, 7)
            indices = torch.empty((2, length, 9), device="meta", dtype=torch.int64)
            for fallback in (False, True):
                actual = sanitize(indices, plan, fill_empty_rows=fallback)
                self.assertEqual(actual.shape, (2, length, 9))
                self.assertEqual(actual.dtype, torch.int32)
                self.assertEqual(actual.device.type, "meta")
    def test_sanitize_does_not_read_tensor_values_on_host(self):
        """Exercise real tensor math while rejecting Python value extraction.

        This is a regression guard for the helper, not proof about NPU runtime
        synchronization or the surrounding attention/communication pipeline.
        """
        plan = Plan(2, 8, 32, 0)
        values = make_values(2, 8, 5, 8, seed=24)
        indices = torch.tensor(values, dtype=torch.int64)
        expected = torch.tensor(reference_indices(values, plan, True), dtype=torch.int32)
        with ExitStack() as stack:
            for name in ("item", "cpu", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__"):
                stack.enter_context(mock.patch.object(
                    torch.Tensor, name, side_effect=AssertionError("host extraction: " + name)
                ))
            stack.enter_context(mock.patch.object(
                torch, "equal", side_effect=AssertionError("host-result tensor equality")
            ))
            actual = sanitize(indices, plan, fill_empty_rows=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
