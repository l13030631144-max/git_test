# vLLM 学习教程

从零开始学习 vLLM 的完整教程，涵盖基础原理到源码解读，包含大量可运行的代码示例。

## 项目结构

```
vllm_learn/
├── vllm_learn1.py              # 第一阶段：前置知识
├── vllm_learn2.py              # 第二阶段：核心原理
├── vllm_learn3.py              # 第三阶段：基础实践
├── vllm_learn4.py              # 第四阶段：进阶使用
├── vllm_learn5.py              # 第五阶段：源码解读
├── Qwen2.5-0.5B-Instruct/     # 本地模型（用于实践练习）
├── dataset_lora_agent/         # LoRA 微调数据集
├── lora_adapter/               # 训练好的 LoRA 适配器权重
└── README.md
```

## 学习路线

### 第一阶段：前置知识（vllm_learn1.py）

- LLM 推理的基本流程（自回归生成）
- KV Cache 的概念与显存估算
- 推理瓶颈分析：显存浪费、批处理效率低、显存碎片化

### 第二阶段：核心原理（vllm_learn2.py）

- **PagedAttention**：借鉴操作系统虚拟内存分页，按需管理 KV Cache
- **Continuous Batching**：请求随到随走，GPU 始终满载
- vLLM 整体架构概览

### 第三阶段：基础实践（vllm_learn3.py）

- 安装 vLLM
- 离线推理（Offline Inference）
- 采样参数详解（temperature、top_p、top_k 等）
- Chat 对话模式（单轮/多轮）
- 启动 OpenAI 兼容 API 服务
- 批量推理与性能测试
- 关键参数调优指南

### 第四阶段：进阶使用（vllm_learn4.py）

- 多卡推理（Tensor Parallel）
- 流式输出（Streaming）
- LoRA 适配器动态加载
- 量化模型推理（AWQ / GPTQ / FP8）
- 前缀缓存（Prefix Caching）
- 结构化输出（Guided Decoding）
- 多模态模型推理

### 第五阶段：源码解读（vllm_learn5.py）

- vLLM 源码目录结构
- 核心调用链（一次推理请求的完整旅程）
- Scheduler 调度器详解
- Block Manager 块管理器详解
- ModelRunner 模型执行器
- 推荐的源码阅读顺序与调试技巧

## 环境要求

- Python 3.9 ~ 3.12
- NVIDIA GPU（推荐 >= 16GB 显存）
- CUDA 环境

## 快速开始

```bash
# 安装 vLLM
pip install vllm

# 从第一阶段开始学习（无需 GPU）
python vllm_learn1.py

# 第三阶段开始需要 GPU
python vllm_learn3.py
```

## 建议

- **第 1~2 阶段**：先运行代码看效果，建立直觉（无需 GPU）
- **第 3 阶段**：动手实践，每个练习都跑一遍
- **第 4 阶段**：按需学习，用到哪个学哪个
- **第 5 阶段**：有余力再深入，不强求
