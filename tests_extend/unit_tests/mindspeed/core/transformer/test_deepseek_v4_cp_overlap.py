# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU-only contract tests for the real CP attention orchestration methods.

The methods are loaded from their AST so these scheduling checks can run without
importing torch_npu or Megatron. Tensor arithmetic, HCCL completion semantics and
numerical/gradient parity still require the distributed NPU integration tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ATTENTION_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "mindspeed/core/transformer/deepseek_v4/attention.py"
)


class Tensor:
    """Preserve tensor provenance across the layout-only operations under test."""

    def __init__(self, name, *, detached=False, shape=(8, 1, 2, 4)):
        self.name = name
        self.detached = detached
        self.shape = shape

    def detach(self):
        return Tensor(self.name, detached=True, shape=self.shape)

    def transpose(self, *_):
        return self

    def contiguous(self):
        return self

    def reshape(self, *_):
        return self

    def dim(self):
        return len(self.shape)


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def replace(obj, **changes):
    return SimpleNamespace(**(vars(obj) | changes))


class ScheduleHarness:
    def __init__(self, case, *, ratio=4, bsnd=True, fused=True, loss=False, indexer=True, identity=False):
        self.case = case
        self.events = []
        self.pending = {}
        self.inputs = {}
        self.loss = loss
        self.metadata = SimpleNamespace(is_identity_compact_order=identity)
        # Internal padding deliberately prevents using raw, un-compacted starts.
        self.raw_starts = (0, -1, 4, -1)
        self.valid_starts = (0, 4)
        self.compact_starts = {}
        self.args = SimpleNamespace(fp8=None, use_fused_lightning_indexer=fused, indexer_loss_coeff=0.5)
        self.runtime = SimpleNamespace(
            is_bsnd=bsnd, layout="BSND" if bsnd else "TND", seq_dim=1 if bsnd else 0,
            query=Tensor("q"), ori_kv=Tensor("ori"), window_size=4,
            cp_group="cp", cp_global_ranks=(0, 1), cu_seqlens_global=(0, 16),
            local_seq_offset=0, query_positions=tuple(range(8)), batch_size=1, seq_len=8,
            cu_seqlens_q=(0, 8), shared_query_positions=lambda: tuple(range(8)),
            shared_cu_seqlens_global=lambda: (0, 16),
        )
        self.candidates = SimpleNamespace(
            candidate_blocks=Tensor("candidate"), metadata=self.metadata,
            selected_indices=(0, 2), output_size=4, cp_group="cp",
            compress_context=SimpleNamespace(
                valid_mask=(True, False, True, False),
                left_context_blocks=Tensor("left"),
                left_context_boundary_blocks=Tensor("boundary"),
            ),
        )
        globals_ = self._globals()
        tree = ast.parse(ATTENTION_SOURCE.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepSeekV4SelfAttentionCP")
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"_prepare_compressed_kv", "_run_cp_attention"}]
        case.assertEqual(len(methods), 2)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(ATTENTION_SOURCE), "exec"), globals_)
        subject_type = type("Subject", (), {node.name: globals_[node.name] for node in methods})
        self.subject = subject_type()
        self.subject.config = SimpleNamespace()
        self.subject.compress_ratio = ratio
        self.subject.window_size = 4
        self.subject.mtp_idx = 0
        self.subject.compressor = "main"
        self.subject.indexer = SimpleNamespace(
            kv_compressor="indexer", index_topk=2, forward_query_weights=self.query_weights,
        ) if indexer else None
        self.subject.attn_sink = Tensor("sink")
        self.subject.softmax_scale = 0.5
        self.subject.indexer_loss_tracker = object()
        self.subject._windowed_ori_kv = self.window

    def _globals(self):
        harness = self

        class Compressor:
            def __init__(self, name, *_args, prepared_candidate=None):
                self.name = name
                self.prepared_candidate = prepared_candidate

            def compress(self, blocks, context):
                harness.events.append("compress_" + self.name)
                detached = self.name == "indexer"
                harness.case.assertEqual(blocks.detached, detached)
                if self.prepared_candidate is not None:
                    harness.case.assertEqual(self.prepared_candidate.detached, detached)
                harness.case.assertEqual(context.left_context_blocks.detached, detached)
                harness.case.assertEqual(context.left_context_boundary_blocks.detached, detached)
                return Tensor(self.name)

        return {
            "get_args": lambda: self.args, "replace": replace,
            "torch": SimpleNamespace(no_grad=NoGrad, ones_like=lambda *_args, **_kwargs: (True, True), bool=bool),
            "_build_runtime": lambda *_args, **_kwargs: self.runtime,
            "exchange_deepseek_v4_previous_tails": lambda **_kwargs: (Tensor("previous_ori"), Tensor("previous_source")),
            "prepare_deepseek_v4_compression_candidates_for_cp": lambda *_args, **_kwargs: self.candidates,
            "_prepare_shared_fp32_candidate": lambda *_args: Tensor("shared"),
            "_CPCompressor": Compressor,
            "launch_deepseek_v4_allgather_compressed_kv": self.launch,
            "wait_deepseek_v4_compressed_kv": self.wait,
            "compact_deepseek_v4_compressed_kv": self.compact,
            "flatten_deepseek_v4_cp_tensor_to_tnd": lambda tensor, **_kwargs: tensor,
            "run_deepseek_v4_right_aligned_fused_indexer": self.fused_topk,
            "finalize_deepseek_v4_fused_indexer_compact_indices": self.finalize_topk,
            "build_deepseek_v4_dense_indexer_compact_indices": self.dense_topk,
            "build_deepseek_v4_causal_cmp_sparse_indices": self.fallback_topk,
            "should_use_deepseek_v4_cp_indexer_loss": lambda *_args: self.loss,
            "_as_float32": lambda tensor: tensor,
            "build_deepseek_v4_owned_runtime_metadata": lambda **kwargs: kwargs,
            "DeepSeekV4CPContextParallel": lambda **_kwargs: SimpleNamespace(forward=self.attention),
        }

    def launch(self, candidates, local):
        self.events.append("launch_" + local.name)
        self.case.assertIs(candidates.metadata, self.metadata)
        self.case.assertEqual(candidates.selected_indices, (0, 2))
        pending = SimpleNamespace(name=local.name, waited=False, metadata=candidates.metadata)
        self.pending[local.name] = pending
        return pending

    def wait(self, pending):
        self.case.assertFalse(pending.waited, "Each AG should be resolved once")
        self.events.append("wait_" + pending.name)
        pending.waited = True
        return pending

    def compact(self, prepared, **_kwargs):
        self.case.assertTrue(prepared.waited, "Gathered values must not be consumed before wait")
        self.events.append("compact_" + prepared.name)
        starts = tuple(value for value in self.raw_starts if value >= 0)
        self.compact_starts[prepared.name] = starts
        return Tensor(prepared.name + "_key"), starts

    def query_weights(self, hidden, query_lora, _freqs):
        self.events.append("query_weights")
        self.case.assertTrue(hidden.detached)
        self.case.assertTrue(query_lora.detached)
        self.case.assertFalse(any(pending.waited for pending in self.pending.values()))
        self.query_index = Tensor("query_index")
        self.weights = Tensor("weights")
        return self.query_index, self.weights

    def _topk(self, key):
        self.case.assertEqual(key.name, "indexer_key")
        self.case.assertTrue(self.pending["indexer"].waited)
        self.case.assertFalse(self.pending["main"].waited, "Top-K must precede the main-KV wait")
        self.events.append("topk")
        return Tensor("topk")

    def fused_topk(self, _indexer, _hidden, _query, key, _weights, *_args, **kwargs):
        self.case.assertEqual(kwargs["identity_single_sample"], self.metadata.is_identity_compact_order and self.runtime.is_bsnd)
        return self._topk(key)

    def finalize_topk(self, indices, starts, *_args, **kwargs):
        self.case.assertIs(starts, self.compact_starts["indexer"])
        self.case.assertEqual(starts, self.valid_starts)
        self.case.assertEqual(kwargs["identity_compact_order"], self.metadata.is_identity_compact_order and self.runtime.is_bsnd)
        return indices, True

    def dense_topk(self, _query, key, _weights, _positions, starts, *_args, **_kwargs):
        self.case.assertIs(starts, self.compact_starts["indexer"])
        self.case.assertEqual(starts, self.valid_starts)
        return self._topk(key)

    def fallback_topk(self, _positions, starts, *_args, **_kwargs):
        self.case.assertTrue(self.pending["main"].waited)
        self.case.assertIs(starts, self.compact_starts["main"])
        self.events.append("fallback_topk")
        return Tensor("fallback_topk")

    def window(self, *_args):
        self.events.append("window")
        return Tensor("window"), (0, 8)

    def attention(self, **kwargs):
        self.case.assertTrue(all(pending.waited for pending in self.pending.values()))
        self.events.append("smla")
        self.inputs = kwargs
        if self.subject.compress_ratio > 1:
            self.case.assertIs(kwargs["prepared_compressed_kv"], self.pending["main"])
            self.case.assertEqual(kwargs["compacted_compressed_kv"].name, "main_key")
            self.case.assertIs(kwargs["compacted_block_starts"], self.compact_starts["main"])
        if self.loss:
            self.case.assertIs(kwargs["query_index"], self.query_index)
            self.case.assertIs(kwargs["weights"], self.weights)
            self.case.assertEqual(kwargs["key_index"].name, "indexer_key")
            self.case.assertIs(kwargs["loss_tracker"], self.subject.indexer_loss_tracker)
        else:
            self.case.assertIsNone(kwargs["query_index"])
            self.case.assertIsNone(kwargs["key_index"])
            self.case.assertIsNone(kwargs["weights"])
        return Tensor("output")

    def run(self):
        return self.subject._run_cp_attention(
            Tensor("q"), Tensor("kv"), Tensor("hidden"), Tensor("query_lora"),
            Tensor("global_freqs"), Tensor("local_freqs"),
            None if self.runtime.is_bsnd else SimpleNamespace(),
        )


class TestDeepSeekV4CPOverlap(unittest.TestCase):
    def test_csa_launch_wait_and_topk_order(self):
        for bsnd in (True, False):
            for fused in (True, False):
                for loss in (True, False):
                    with self.subTest(bsnd=bsnd, fused=fused, loss=loss):
                        h = ScheduleHarness(self, bsnd=bsnd, fused=fused, loss=loss)
                        h.run()
                        expected = [
                            "compress_indexer", "launch_indexer", "compress_main", "launch_main",
                            "query_weights", "wait_indexer", "compact_indexer", "topk",
                            "wait_main", "compact_main", "window", "smla",
                        ]
                        self.assertEqual(h.events, expected)

    def test_fused_identity_compact_order(self):
        h = ScheduleHarness(self, identity=True)
        h.run()
        self.assertTrue(h.inputs["cmp_sparse_indices_are_causal"])

    def test_hca_and_ratio_four_without_indexer(self):
        for ratio in (4, 128):
            for bsnd in (True, False):
                with self.subTest(ratio=ratio, bsnd=bsnd):
                    h = ScheduleHarness(self, ratio=ratio, indexer=False, bsnd=bsnd)
                    h.run()
                    self.assertNotIn("indexer", h.pending)
                    self.assertEqual(h.events[:2], ["compress_main", "launch_main"])
                    self.assertEqual(h.events[-1], "smla")
                    self.assertEqual("fallback_topk" in h.events, ratio == 4)

    def test_swa_has_no_compressed_collective(self):
        for bsnd in (True, False):
            with self.subTest(bsnd=bsnd):
                h = ScheduleHarness(self, ratio=1, indexer=False, bsnd=bsnd)
                h.run()
                self.assertEqual(h.events, ["window", "smla"])
                self.assertIsNone(h.inputs["prepared_compressed_kv"])
                self.assertIsNone(h.inputs["cmp_sparse_indices"])


if __name__ == "__main__":
    unittest.main()
