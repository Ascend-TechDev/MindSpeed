# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU-only contract tests for the real CP attention orchestration methods.

The methods are loaded from their AST so these scheduling checks can run without
importing torch_npu or Megatron. Tensor arithmetic, HCCL completion semantics and
numerical/gradient parity still require the distributed NPU integration tests.
"""

import argparse
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

    def __init__(self, name, *, detached=False, shape=(8, 1, 2, 4), npu=None, device=None):
        self.name = name
        self.detached = detached
        self.shape = shape
        self.npu = npu
        self.device = device or SimpleNamespace(type="npu", index=0)

    def detach(self):
        return Tensor(self.name, detached=True, shape=self.shape, npu=self.npu, device=self.device)

    def record_stream(self, stream):
        self.npu.tensor_records.append((self, stream))

    def transpose(self, *_):
        return self

    def contiguous(self):
        return self

    def reshape(self, *_):
        return self

    def dim(self):
        return len(self.shape)


class NoGrad:
    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        self.previous = self.torch.grad_enabled
        self.torch.grad_enabled = False
        return self

    def __exit__(self, *_):
        self.torch.grad_enabled = self.previous
        return False


class Stream:
    def __init__(self, npu, device, name):
        self.npu = npu
        self.device = device
        self.name = name

    def wait_stream(self, producer):
        self.npu.case.assertIs(producer, self.npu.main)
        self.npu.case.assertIsNot(self, producer)
        # The dependency must be captured before main Compressor is enqueued.
        self.npu.case.assertEqual(self.npu.harness.events[-1], "launch_indexer")
        self.npu.harness.events.append("side_wait_producer")
        self.npu.producer_waits.append((self, producer))

    def wait_event(self, event):
        self.npu.case.assertIs(self, self.npu.main)
        self.npu.case.assertIsNotNone(event.recorded_stream)
        self.npu.case.assertEqual(self.npu.harness.events[-1], "compact_indexer")
        self.npu.harness.events.append("wait_query_done")
        self.npu.joined_events.append(event)


class Event:
    def __init__(self, npu):
        self.npu = npu
        self.recorded_stream = None

    def record(self, stream):
        self.npu.case.assertIs(stream, self.npu.current)
        self.npu.case.assertIsNot(stream, self.npu.main)
        self.recorded_stream = stream
        self.npu.harness.events.append("record_query_done")


class StreamContext:
    def __init__(self, npu, stream):
        self.npu = npu
        self.stream = stream

    def __enter__(self):
        self.previous = self.npu.current
        self.npu.current = self.stream
        return self

    def __exit__(self, *_):
        self.npu.current = self.previous
        return False


class NPU:
    """Record dependencies only; this fake does not emulate asynchronous execution."""

    def __init__(self, harness):
        self.harness = harness
        self.case = harness.case
        self.main = Stream(self, SimpleNamespace(type="npu", index=0), "main")
        self.current = self.main
        self.streams = []
        self.events = []
        self.producer_waits = []
        self.joined_events = []
        self.tensor_records = []

    def current_stream(self, device=None):
        if device is not None:
            self.case.assertEqual(self.current.device, device)
        return self.current

    def Stream(self, *, device):
        stream = Stream(self, device, "indexer_" + str(len(self.streams)))
        self.streams.append(stream)
        return stream

    def Event(self):
        event = Event(self)
        self.events.append(event)
        return event

    def stream(self, stream):
        return StreamContext(self, stream)


class Torch:
    bool = bool

    def __init__(self, harness):
        self.npu = NPU(harness)
        self.grad_enabled = True

    def no_grad(self):
        return NoGrad(self)

    def is_grad_enabled(self):
        return self.grad_enabled

    @staticmethod
    def ones_like(*_args, **_kwargs):
        return (True, True)


def replace(obj, **changes):
    return SimpleNamespace(**(vars(obj) | changes))


class ScheduleHarness:
    def __init__(self, case, *, ratio=4, bsnd=True, fused=True, loss=False,
                 indexer=True, identity=False, overlap=None):
        self.case = case
        self.events = []
        self.pending = {}
        self.inputs = {}
        self.loss = loss
        self.overlap = bool(overlap)
        self.torch = Torch(self)
        self.query_calls = []
        self.metadata = SimpleNamespace(is_identity_compact_order=identity)
        # Internal padding deliberately prevents using raw, un-compacted starts.
        self.raw_starts = (0, -1, 4, -1)
        self.valid_starts = (0, 4)
        self.compact_starts = {}
        self.args = SimpleNamespace(fp8=None, use_fused_lightning_indexer=fused, indexer_loss_coeff=0.5)
        if overlap is not None:
            self.args.deepseek_v4_cp_indexer_query_overlap = overlap
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
        method_names = {"_prepare_compressed_kv", "_run_cp_attention",
                        "_launch_indexer_query_weights", "_wait_indexer_query_weights"}
        methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in method_names]
        case.assertEqual(len(methods), len(method_names))
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
        self.subject._indexer_query_weights_stream = None

    def _globals(self):
        harness = self

        class Compressor:
            def __init__(self, name, *_args, prepared_candidate=None):
                self.name = name
                self.prepared_candidate = prepared_candidate

            def compress(self, blocks, context):
                harness.case.assertIs(harness.torch.npu.current, harness.torch.npu.main)
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
            "torch": self.torch,
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
        self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
        self.events.append("launch_" + local.name)
        self.case.assertIs(candidates.metadata, self.metadata)
        self.case.assertEqual(candidates.selected_indices, (0, 2))
        pending = SimpleNamespace(name=local.name, waited=False, metadata=candidates.metadata)
        self.pending[local.name] = pending
        return pending

    def wait(self, pending):
        self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
        self.case.assertFalse(pending.waited, "Each AG should be resolved once")
        self.events.append("wait_" + pending.name)
        pending.waited = True
        return pending

    def compact(self, prepared, **_kwargs):
        self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
        self.case.assertTrue(prepared.waited, "Gathered values must not be consumed before wait")
        self.events.append("compact_" + prepared.name)
        starts = tuple(value for value in self.raw_starts if value >= 0)
        self.compact_starts[prepared.name] = starts
        return Tensor(prepared.name + "_key"), starts

    def query_weights(self, hidden, query_lora, freqs):
        self.events.append("query_weights")
        self.case.assertTrue(hidden.detached)
        self.case.assertTrue(query_lora.detached)
        self.case.assertFalse(any(pending.waited for pending in self.pending.values()))
        self.case.assertIs(freqs, self.run_inputs["local_freqs"])
        self.case.assertFalse(freqs.detached)
        if self.overlap:
            self.case.assertIs(self.torch.npu.current, self.subject._indexer_query_weights_stream)
            self.case.assertNotIn("main", self.pending, "Query work must launch before main KV AG")
        else:
            self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
        self.query_index = Tensor("query_index", npu=self.torch.npu, device=hidden.device)
        self.weights = Tensor("weights", npu=self.torch.npu, device=hidden.device)
        self.query_calls.append((hidden, query_lora, freqs, self.torch.is_grad_enabled(),
                                 self.query_index, self.weights))
        return self.query_index, self.weights

    def _topk(self, key):
        self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
        if self.overlap:
            self.case.assertIs(self.torch.npu.joined_events[-1], self.torch.npu.events[-1])
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
        self.case.assertIs(self.torch.npu.current, self.torch.npu.main)
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
        self.pending = {}
        self.compact_starts = {}
        self.run_inputs = {
            name: Tensor(name, npu=self.torch.npu, device=self.torch.npu.main.device)
            for name in ("q", "kv", "hidden", "query_lora", "global_freqs", "local_freqs")
        }
        return self.subject._run_cp_attention(
            *(self.run_inputs[name] for name in
              ("q", "kv", "hidden", "query_lora", "global_freqs", "local_freqs")),
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

    def test_side_stream_launch_and_join_order(self):
        for bsnd in (True, False):
            for fused in (True, False):
                for loss in (True, False):
                    with self.subTest(bsnd=bsnd, fused=fused, loss=loss):
                        h = ScheduleHarness(self, bsnd=bsnd, fused=fused, loss=loss, overlap=True)
                        h.run()
                        self.assertEqual(h.events, [
                            "compress_indexer", "launch_indexer", "side_wait_producer",
                            "query_weights", "record_query_done", "compress_main", "launch_main",
                            "wait_indexer", "compact_indexer", "wait_query_done", "topk",
                            "wait_main", "compact_main", "window", "smla",
                        ])
                        self.assertIs(h.torch.npu.current, h.torch.npu.main)
                        self.assertEqual(len(h.torch.npu.producer_waits), 1)
                        self.assertEqual(len(h.torch.npu.joined_events), 1)

    def test_side_stream_tensor_lifetimes(self):
        h = ScheduleHarness(self, overlap=True, loss=True)
        h.run()
        records = h.torch.npu.tensor_records
        self.assertEqual([tensor.name for tensor, _ in records],
                         ["hidden", "query_lora", "local_freqs", "query_index", "weights"])
        side = h.subject._indexer_query_weights_stream
        self.assertTrue(all(stream is side for _, stream in records[:3]))
        self.assertTrue(all(stream is h.torch.npu.main for _, stream in records[3:]))
        hidden, query_lora, freqs, _, query_index, weights = h.query_calls[0]
        self.assertEqual([tensor for tensor, _ in records],
                         [hidden, query_lora, freqs, query_index, weights])
        self.assertTrue(hidden.detached)
        self.assertTrue(query_lora.detached)
        self.assertIs(freqs, h.run_inputs["local_freqs"])
        self.assertFalse(h.run_inputs["hidden"].detached)
        self.assertFalse(h.run_inputs["query_lora"].detached)

    def test_checkpoint_grad_mode_and_fresh_results(self):
        h = ScheduleHarness(self, overlap=True, loss=True)
        # Reentrant checkpoint's first pass uses no_grad; recomputation enables it.
        with h.torch.no_grad():
            h.run()
            self.assertFalse(h.torch.is_grad_enabled())
        first_stream = h.subject._indexer_query_weights_stream
        first_inputs = h.run_inputs
        h.run()
        self.assertTrue(h.torch.is_grad_enabled())
        self.assertEqual([call[3] for call in h.query_calls], [False, True])
        self.assertIs(h.subject._indexer_query_weights_stream, first_stream)
        self.assertEqual(len(h.torch.npu.streams), 1)
        self.assertEqual(len(h.torch.npu.events), 2)
        self.assertIsNot(h.torch.npu.events[0], h.torch.npu.events[1])
        self.assertEqual(h.torch.npu.joined_events, h.torch.npu.events)
        self.assertIsNot(h.query_calls[0][4], h.query_calls[1][4])
        self.assertIsNot(h.query_calls[0][5], h.query_calls[1][5])
        self.assertIsNot(first_inputs["local_freqs"], h.run_inputs["local_freqs"])

    def test_side_stream_is_recreated_on_device_change(self):
        h = ScheduleHarness(self, overlap=True)
        h.run()
        first_stream = h.subject._indexer_query_weights_stream
        h.torch.npu.main.device = SimpleNamespace(type="npu", index=1)
        h.run()
        self.assertEqual(len(h.torch.npu.streams), 2)
        self.assertIsNot(h.subject._indexer_query_weights_stream, first_stream)
        self.assertEqual(h.subject._indexer_query_weights_stream.device.index, 1)

    def test_explicitly_disabled_overlap_does_not_create_stream(self):
        h = ScheduleHarness(self, overlap=False)
        h.run()
        self.assertEqual(h.torch.npu.streams, [])
        self.assertEqual(h.torch.npu.events, [])
        self.assertEqual(h.torch.npu.tensor_records, [])
        self.assertLess(h.events.index("launch_main"), h.events.index("query_weights"))

    def test_overlap_argument_is_opt_in_in_both_entrypoints(self):
        root = ATTENTION_SOURCE.parents[4]
        for relative in ("mindspeed/arguments.py",
                         "mindspeed/features_manager/context_parallel/context_parallel_feature.py"):
            with self.subTest(path=relative):
                source = root / relative
                tree = ast.parse(source.read_text(encoding="utf-8"))
                registrations = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--deepseek-v4-cp-indexer-query-overlap"
                ]
                self.assertEqual(len(registrations), 1)
                parser = argparse.ArgumentParser()
                parser.add_argument(
                    *(ast.literal_eval(arg) for arg in registrations[0].args),
                    **{kw.arg: ast.literal_eval(kw.value) for kw in registrations[0].keywords},
                )
                self.assertFalse(parser.parse_args([]).deepseek_v4_cp_indexer_query_overlap)
                self.assertTrue(parser.parse_args(
                    ["--deepseek-v4-cp-indexer-query-overlap"]
                ).deepseek_v4_cp_indexer_query_overlap)

    def test_overlap_argument_guards_in_both_entrypoints(self):
        root = ATTENTION_SOURCE.parents[4]
        cases = [
            (True, "deepseek_v4_cp_algo", 8, None, None),
            (True, "deepseek_v4_cp_algo", 8, False, None),
            (True, "deepseek_v4_cp_algo", 8, "hybrid", "FP8"),
            (True, "deepseek_v4_cp_algo", 1, None, "greater than 1"),
            (True, "megatron_cp_algo", 8, None, "deepseek_v4_cp_algo"),
            (False, "megatron_cp_algo", 1, "hybrid", None),
        ]
        for relative in ("mindspeed/arguments.py",
                         "mindspeed/features_manager/context_parallel/context_parallel_feature.py"):
            source = root / relative
            tree = ast.parse(source.read_text(encoding="utf-8"))
            guards = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Name) and node.test.func.id == "getattr"
                and len(node.test.args) > 1 and isinstance(node.test.args[1], ast.Constant)
                and node.test.args[1].value == "deepseek_v4_cp_indexer_query_overlap"
            ]
            self.assertEqual(len(guards), 1)
            compiled = compile(ast.Module(body=guards, type_ignores=[]), str(source), "exec")
            for enabled, algorithm, size, fp8, error in cases:
                with self.subTest(path=relative, enabled=enabled, algorithm=algorithm, size=size, fp8=fp8):
                    args = SimpleNamespace(deepseek_v4_cp_indexer_query_overlap=enabled,
                                           context_parallel_algo=algorithm,
                                           context_parallel_size=size, fp8=fp8)
                    if error is None:
                        exec(compiled, {"args": args})
                    else:
                        with self.assertRaisesRegex(AssertionError, error):
                            exec(compiled, {"args": args})

    def test_fused_identity_compact_order(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                h = ScheduleHarness(self, identity=True, overlap=overlap)
                h.run()
                self.assertTrue(h.inputs["cmp_sparse_indices_are_causal"])

    def test_hca_and_ratio_four_without_indexer(self):
        for ratio in (4, 128):
            for bsnd in (True, False):
                with self.subTest(ratio=ratio, bsnd=bsnd):
                    h = ScheduleHarness(self, ratio=ratio, indexer=False, bsnd=bsnd, overlap=True)
                    h.run()
                    self.assertNotIn("indexer", h.pending)
                    self.assertEqual(h.events[:2], ["compress_main", "launch_main"])
                    self.assertEqual(h.events[-1], "smla")
                    self.assertEqual("fallback_topk" in h.events, ratio == 4)
                    self.assertEqual(h.torch.npu.streams, [])
                    self.assertEqual(h.torch.npu.events, [])

    def test_swa_has_no_compressed_collective(self):
        for bsnd in (True, False):
            with self.subTest(bsnd=bsnd):
                h = ScheduleHarness(self, ratio=1, indexer=False, bsnd=bsnd, overlap=True)
                h.run()
                self.assertEqual(h.events, ["window", "smla"])
                self.assertIsNone(h.inputs["prepared_compressed_kv"])
                self.assertIsNone(h.inputs["cmp_sparse_indices"])
                self.assertEqual(h.torch.npu.streams, [])


if __name__ == "__main__":
    unittest.main()
