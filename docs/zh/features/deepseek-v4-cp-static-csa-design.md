# DeepSeek-V4 CSA CP 静态布局与设备端索引准备

## 背景与挑战

DeepSeek-V4 CSA 在长序列并行训练中，需要分别收集 Indexer Key 和主 Attention 的压缩 KV。即使 AllGather 已与计算重叠，Indexer 和 SparseFlashMla 之间的布局准备仍可能引入 Host 同步，影响设备计算的连续性。

现有路径主要存在以下重复准备开销：

- Compressor 准备时，将设备上的 `cu_seqlens` 读回 Host，以构造压缩块计划。
- Indexer 前，通过 `.item()`、动态 `nonzero` 和 `.tolist()` 从设备上的长度、位置反推 Query 分段，再由 Python 逐段调用算子。
- Indexer 包装器将设备上的 residual 读回 Host，以生成 metadata 缓存键。
- Top-K 后，再次校验 BSND 样本长度和位置连续性，动态选择压缩前缀，并构造 SparseFlashMla 与 Indexer loss 的输入布局。

在固定长度 BSND 场景中，上述长度、起点和前缀容量可以由输入形状及 CP/TP 拓扑直接确定。重复从设备获取这些已知信息，会造成额外的同步和算子下发间隙。profiling 中的 FREE 表示统计时间轴上未被计算或通信任务覆盖的空洞，不能直接等同于内存释放开销，也不能全部归因于这些布局操作。

## 解决方案

参考 Megatron 原生 CSA 的固定容量、设备端索引准备思路，在现有 NPU 路径中增加可选的静态 BSND 分支。参考版本为 `3630e44a8046c175f3b6f399204755892b84369a`，主要参考 `csa_utils/packed_layout.py` 的 `_build_attention_indices`、`_sanitize_topk`，以及 `csa_utils/cp_utils.py` 的布局构造方式。

**1. 在元数据生产端构造 Host 计划。**

Attention 在创建设备端 `cu_seqlens`、`query_positions` 的同时，使用 Python 整数创建不可变的 `DeepSeekV4CPStaticCSAPlan`，保存 batch 大小、Query 长度、全局序列长度、Query 起点和压缩比。每次前向及重计算重新构造计划，不缓存激活或设备张量。

设每个 CP rank 的完整局部 KV 长度为 `L_kv`，CP 大小为 `C`，rank 为 `r`，本地 Query 长度为 `L_q`，压缩比为 4：

```text
global_length = C * L_kv
query_start = r * L_kv + tp_query_offset
query_end = query_start + L_q
global_blocks = floor(global_length / 4)
prefix_blocks = floor(query_end / 4)
residual = query_end % 4
```

Query 已恢复完整局部序列或未启用 SP 时，`tp_query_offset=0`；SP 下若 `L_kv=TP_size*L_q` 且 Query 仍按 TP 切分，则 `tp_query_offset=tp_rank*L_q`。因此，Query 起点不能直接替换为 KV 分片起点。

Compressor 接收计划生成的 Host `cu_seqlens` tuple，避免为获取相同边界先进行设备到 Host 的传输。多 batch 场景下，该 tuple 保留完整的 `B+1` 个边界。

**2. 直接使用真实 Key 前缀与 Host residual 执行 Indexer。**

静态 BSND 路径只有一个连续 Query 段，可直接按 `prefix_blocks` 裁剪 Indexer Key，并将 `(residual,) * B` 传入包装器。包装器使用 Host 值构造既有 metadata 缓存键，同时生成 CANN 所需的 int32 设备 residual tensor。

Ascend BSND Lightning Indexer 根据实际 Key 长度和 residual 恢复右下角因果对齐。因此仍保留真实 prefix 切片，不能将全局 AllGather 缓冲区的 padding 当成有效 Key。本方案中的固定容量，是指容量和切片边界由输入形状与并行布局确定，不是设备内存池复用，也不要求各 CP rank 的前缀长度相同。

**3. 在设备端完成固定宽度的索引准备。**

`sanitize_static_csa_indices()` 保持 `[B, L_q, TopK]` 形状，通过设备端比较、`where` 和 reduction 过滤未来、负数及越界索引。对于第 `i` 行 Query：

```text
visible[i] = floor((query_start + i + 1) / 4)
valid[b,i,j] = (index[b,i,j] >= 0)
             & (index[b,i,j] < visible[i])
             & (index[b,i,j] < global_blocks)
```

无效槽位设为 `-1`。保留现有 SparseFlashMla fallback：一行没有有效选择、但存在因果可见块时，将第一列填为 0；没有可见块时保持全 `-1`。大 int64 索引先屏蔽再转换为 int32，避免整数回绕；CANN 返回的 int32 索引不额外转换为完整的 int64 Top-K 缓冲区。

在 identity compact order 下，样本内逻辑压缩块编号等于物理 compact 编号，前缀从 0 开始裁剪后编号不变。准备阶段复杂度为 `O(B * L_q * TopK + global_blocks)`，不需要构造全量 Query-Key visibility 矩阵。原 identity 路径也可能跳过该矩阵校验，因此主要收益目标是减少重复 Host 同步，而非假设所有输入都节省一份完整矩阵。

**4. 统一 Attention 和 Indexer loss 的前缀描述。**

SparseFlashMla 与 Indexer loss Key 共用 `DeepSeekV4CPAlignmentDescriptor`，通过现有 `narrow(...).contiguous()` 路径裁剪，保留 autograd 边。梯度回写至全局 compact buffer 的对应前缀，其余行梯度为 0，再沿既有 CP 反向路径传播。SparseFlashMlaGrad、Indexer KL loss 和 ReduceScatter 的接口及调度保持原实现。

优化前后的关键流程如下：

```mermaid
flowchart TD
    subgraph Before[原路径]
        A1[1. 创建设备长度与位置] --> A2[2. 读回长度并动态构造分段]
        A2 --> A3[3. plan 和 residual 回 Host]
        A3 --> A4[4. Lightning Indexer / Top-K]
        A4 --> A5[5. 重建并验证 SMLA 前缀]
        A5 --> A6[6. 索引整理与 SparseFlashMla]
    end
    subgraph After[静态 BSND 路径]
        B1[1. 形状及拓扑生成 Host plan] --> B2[2. 直接确定 prefix 和 residual]
        B2 --> B3[3. Lightning Indexer / Top-K]
        B3 --> B4[4. 设备端索引 mask 与 fallback]
        B1 -. 相同前缀描述 .-> B5[5. SparseFlashMla 与 Indexer loss Key 对齐]
        B4 --> B5
    end
```

图中省略了保持原顺序的通信操作：Indexer Key AG 先发，主 KV AG 的等待仍位于 Top-K 之后。可选 Query/weights 侧流的生产依赖、完成事件及 tensor 生命周期保持原实现。

主要代码入口如下，路径均相对 MindSpeed 根目录：

| 文件 | 功能 |
| --- | --- |
| `mindspeed/core/transformer/deepseek_v4/attention.py` | 生产 Host plan、传递 Host 边界并选择静态路径 |
| `mindspeed/core/transformer/deepseek_v4/cp_indexer.py` | 按静态前缀执行 fused Indexer |
| `mindspeed/core/context_parallel/deepseek_v4_context_parallel/_static_csa.py` | 不可变计划及设备端索引过滤 |
| `mindspeed/core/context_parallel/deepseek_v4_context_parallel/_smla_inputs.py` | 构造 SMLA 前缀及共享 alignment |
| `mindspeed/core/context_parallel/deepseek_v4_context_parallel/ops/npu_lightning_indexer.py` | Host residual 与算子 metadata 对接 |

## 使用场景

适用于固定长度 BSND 的 DeepSeek-V4 CSA CP 训练，进入优化路径需满足以下条件：

- 使用 `deepseek_v4_cp_algo`，CP 大小大于 1，并启用 fused Lightning Indexer。
- CSA 压缩比为 4，模型拥有 Indexer。
- 输入为非 packed BSND，各 batch 样本等长，共享连续的局部 Query 区间。
- 压缩 KV 由现有生产端建立 identity compact order，且 `batch_shared_sequence=True`。

当前 DeepSeek CP 使用等长、连续、按 rank 顺序切分。静态计划兼容 Query 已恢复完整序列及 Query 保留 TP/SP 子区间两种情况。HCA、SWA、dense Indexer、packed TND、非 identity 压缩顺序及没有静态计划的调用继续使用原路径。

此特性适合优化 Indexer 分段、Top-K 后布局准备中的 Host 空洞，不改变通信字节数。Compressor 内部部分位置最大值读取、左上下文动态索引、CANN metadata kernel、内存分配及跨 rank 等待仍可能产生开销。设备端索引准备目前由多个 PyTorch 操作组成，不依赖 `torch.compile`，也不代表已融合成单个 NPU kernel 或已验证图捕获支持。

## 使用方法

| 重要参数 | 参数说明 | 是否必选 | 默认值 |
| --- | --- | --- | --- |
| `--context-parallel-algo deepseek_v4_cp_algo` | 使用 DeepSeek-V4 压缩注意力 CP 路径 | 是 | 依参数入口而定 |
| `--context-parallel-size [int]` | CP 大小，需大于 1 | 是 | `1` |
| `--use-fused-lightning-indexer` | 启用 fused Lightning Indexer | 是 | 关闭 |
| `--deepseek-v4-cp-static-csa` | 启用静态 BSND 计划及设备端索引准备 | 是 | 关闭 |
| `--deepseek-v4-cp-indexer-query-overlap` | 在侧流计算 Indexer Query/weights，可与本特性组合使用 | 否 | 关闭 |

在已配置 CSA 模型和 Indexer 的训练命令中增加：

```bash
--context-parallel-algo deepseek_v4_cp_algo \
--context-parallel-size 8 \
--use-fused-lightning-indexer \
--deepseek-v4-cp-static-csa
```

移除 `--deepseek-v4-cp-static-csa` 即恢复原布局准备路径。进行 A/B 对比时，固定 Query 侧流开关、模型参数、TP/CP/EP、重计算、Indexer loss、数据、随机种子和采集窗口，仅改变本特性开关。

**本地验证方法：**

```bash
python -m unittest discover \
  -s tests_extend/unit_tests/mindspeed/core/transformer \
  -p 'test_deepseek_v4_cp_static_csa*.py' -v

python -m unittest discover \
  -s tests_extend/unit_tests/mindspeed/core/transformer \
  -p 'test_deepseek_v4_cp_overlap.py' -v
```

使用真实 CPU PyTorch 的 24 项新增测试覆盖索引 oracle、CP8/TP2 偏移、多 batch、residual、因果边界、fallback、前缀与裁剪梯度、回退及 metadata 失效；11 项既有测试覆盖通信/Top-K 顺序和侧流调度。CPU 测试隔离了 Megatron、torch_npu 的包初始化，并以 dense 数学 oracle 替代 CANN Indexer 内核，不能代替真实 NPU 算子或多卡训练验证。

**NPU 验证方法：**

| 验证层次 | 配置 | 检查项 |
| --- | --- | --- |
| 单算子 | BSND、B=1/多 B、非零 residual、prefix 小于 Top-K | Lightning Indexer 有效索引、SparseFlashMla 输出与输入梯度 |
| 单层 | CP2/4、TP1/2 | 前向输出、参数及输入梯度、Indexer loss |
| 目标配置 | CP8/TP2、64K、相同 EP 与 batch | 原始前向和完整重计算分别对齐，各 rank collective 顺序一致 |
| 性能 | 相同 warmup、step 和采集窗口 | CSA 总区间、FREE、计算和未覆盖通信区间，以及完整 step 时间 |

profiling 应重点观察 Indexer 启动前的分段读回、residual 的设备到 Host 传输，以及 Top-K 到 SparseFlashMla 之间的前缀校验是否减少。时间统计需按区间并集区分计算、通信与 FREE，不能相加异步 CPU 和设备耗时，也不能仅根据 FREE 百分比判断收益。

本地 CPU 测试已通过；真实 NPU/CANN/HCCL 前后向和性能仍需在目标配置上验收，尚无本特性的实测加速数据。
