# Kimi KDA chunkwise CP 示例

本目录提供模型侧 attention 适配函数。它使用 MindSpeed/Megatron-Core 已创建的 CP group，调用 FLA 的卷积和 KDA CP 接口。完整说明见 [接入文档](../../docs/zh/features/kimi-fla-chunkwise-cp.md)。

本示例随 [MindSpeed #2](https://github.com/Ascend-TechDev/MindSpeed/pull/2) 提交；配套的 Ascend 卷积数值修复与 coregrid 优化见 [FLA #1711](https://github.com/fla-org/flash-linear-attention/pull/1711)。两项 PR 的版本要求、验证范围和性能数据见接入文档。

调用者提供 Kimi attention 子模块、源模型配置和连续 token 分片。该示例不包含完整 Kimi 模型、checkpoint 转换或训练入口，也不会自动替换 MindSpeed 的 attention 实现。

```python
from megatron.core import parallel_state
from examples.kimi_chunkwise_cp.attention import kda_attention_chunkwise

output = kda_attention_chunkwise(
    local_hidden_states,
    attention_module,
    kimi_config,
    parallel_state.get_context_parallel_group(),
)
```

在配置好 PyTorch、torch-npu、FLA 和 HCCL 的环境中，执行两卡验证：

```bash
FLA_CI_ENV=0 ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc-per-node=2 -m pytest \
  tests_extend/unit_tests/mindspeed/core/context_parallel/test_kimi_chunkwise_cp.py -q
```

测试比较 CP2 与 CP1 的 attention 输出、输入梯度和归约后的所有参数梯度，同时覆盖 reentrant checkpoint。测试使用小规模随机权重；真实权重 CP16/256K 的历史结果见接入文档，二者不能互相替代。
