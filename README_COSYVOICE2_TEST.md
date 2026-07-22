# CosyVoice2 C++ vs Python 一致性测试指南

## 概述

本文档说明如何验证 C++ 实现的 CosyVoice2 流式推理与官方 Python 实现的输出一致性。

参考官方实现：https://github.com/FunAudioLLM/CosyVoice.git

## 文件结构

```
/workspace/
├── test_cosyvoice2_consistency.py   # Python 对比测试脚本
├── cosyvoice2_export.cpp            # C++ 输出导出工具
├── include/cosyvoice2_model.h       # CosyVoice2 模型定义
└── README_COSYVOICE2_TEST.md        # 本文档
```

## 主要差异 (CosyVoice2 vs CosyVoice3)

根据官方代码分析，主要区别如下：

### 1. Flow 模块
- **CosyVoice2**: 使用 `CausalMaskedDiffWithXvec`
- **CosyVoice3**: 使用 `CausalMaskedDiffWithDiT`

关键差异:
```python
# CosyVoice2 flow.py
class CausalMaskedDiffWithXvec:
    - 使用 pre_lookahead_layer 处理因果卷积
    - token_mel_ratio = 2 (每个 token 生成 2 帧 mel)
    - 支持 streaming=True 的流式推理
```

### 2. Flow Matching
- **CosyVoice2**: `CausalConditionalCFM` 使用固定的随机噪声
```python
set_all_random_seed(0)
self.rand_noise = torch.randn([1, 80, 50 * 300])
```

### 3. LLM 架构
- **CosyVoice2**: 基于 Qwen2 架构，但有不同的 token 集
- Stop tokens 范围：`speech_token_size + 200`

### 4. HiFT Generator
- 两者都使用 `CausalHiFTGenerator`
- 参数配置可能略有不同

## 使用方法

### 步骤 1: 准备环境

```bash
# 克隆官方 CosyVoice 仓库
git clone https://github.com/FunAudioLLM/CosyVoice.git /tmp/CosyVoice
cd /tmp/CosyVoice
pip install -e .

# 设置环境变量
export COSYVOICE_REPO_PATH=/tmp/CosyVoice
```

### 步骤 2: 编译 C++ 导出工具

```bash
cd /workspace
g++ -std=c++17 -I./include -I./temp/work/include \
    cosyvoice2_export.cpp \
    -o cosyvoice2_export \
    -lggml -lcosyvoice
```

### 步骤 3: 导出 C++ 输出

```bash
./cosyvoice2_export \
    --model /path/to/cosyvoice2.gguf \
    --text "你好，这是 CosyVoice2 测试" \
    --output cpp_outputs.bin
```

### 步骤 4: 运行 Python 对比测试

```bash
python3 test_cosyvoice2_consistency.py \
    --model-dir /path/to/model_dir \
    --config /path/to/cosyvoice2.yaml \
    --cpp-outputs cpp_outputs.bin \
    --device cpu \
    --tolerance 1e-3
```

## 测试内容

### 1. LLM Prefill/Decode 测试
对比 LLM 的前向传播输出：
- 输入：token IDs
- 输出：logits
- 容忍度：相对误差 < 1e-4

### 2. Flow Encoder 测试
对比 Mel Spectrogram 生成：
- 输入：speech tokens, prompt tokens, prompt mel, speaker embedding
- 输出：mel spectrogram [1, 80, T]
- 容忍度：相对误差 < 1e-3

### 3. HiFT Generator 测试
对比音频波形生成：
- 输入：mel spectrogram
- 输出：audio waveform
- 容忍度：相对误差 < 1e-2, SNR > 40dB

### 4. 流式推理测试
对比分块处理的累积输出与完整输出的一致性：
- 每个 chunk 的误差 < 5e-3
- 最终输出误差 < 1e-3

## 预期结果

```
============================================================
Running CosyVoice2 Consistency Tests
============================================================

✓ PASSED | llm_prefill
  Max Abs Error: 1.234567e-05
  Max Rel Error: 9.876543e-06
  Details: ✓ Test passed

✓ PASSED | flow_encoder
  Max Abs Error: 2.345678e-04
  Max Rel Error: 1.234567e-04
  Details: ✓ Flow encoder test passed

✓ PASSED | hift_generator
  Max Abs Error: 3.456789e-03
  Max Rel Error: 2.345678e-03
  SNR: 52.34 dB
  Details: ✓ HiFT test passed, SNR: 52.34 dB

============================================================
Summary: 3/3 tests passed
✓ All tests PASSED!
============================================================
```

## 故障排查

### 常见问题

1. **形状不匹配**
   - 检查输入维度是否正确
   - 确认 token_mel_ratio 设置 (应为 2)

2. **误差过大**
   - 检查随机种子是否一致
   - 确认 LayerNorm eps 值 (1e-6)
   - 验证 flow matching 的 t_span 计算

3. **流式推理不一致**
   - 检查 pre_lookahead_len 设置
   - 确认 cache 机制正确实现
   - 验证 chunk 边界处理

## 参考代码位置

官方 Python 实现关键文件：
- `cosyvoice/flow/flow.py`: CausalMaskedDiffWithXvec
- `cosyvoice/flow/flow_matching.py`: CausalConditionalCFM
- `cosyvoice/hift/hift.py`: CausalHiFTGenerator
- `cosyvoice/llm/llm.py`: CosyVoice2LM

C++ 实现关键文件：
- `include/cosyvoice2_model.h`: 模型定义
- `include/cosyvoice_modules.h`: 模块定义
- `src/cosyvoice_loader.cpp`: 模型加载
- `src/cosyvoice_tts.cpp`: TTS 推理流程

## 下一步

1. 完善 `cosyvoice2_model.cpp` 实现
2. 添加流式推理专用接口
3. 优化 KV Cache 管理
4. 添加更多测试用例

