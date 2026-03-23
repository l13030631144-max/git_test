"""
=============================================================
vLLM 学习路线 - 第五阶段：源码解读指引
=============================================================

这一阶段帮你理解 vLLM 的内部架构和源码结构。
不要求你读完所有代码，而是知道"去哪里找什么"。

适合：想深入理解 vLLM / 想给 vLLM 贡献代码 / 想做二次开发的同学。
"""

# =============================================
# 1. vLLM 源码目录结构
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              vLLM 源码目录结构（核心部分）                       ║
╚══════════════════════════════════════════════════════════════════╝

vllm/
├── entrypoints/            ← 【入口层】用户接触的第一层
│   ├── llm.py              ← LLM 类（离线推理入口）⭐ 第一个要看的文件
│   ├── openai/             ← OpenAI 兼容 API 服务
│   │   ├── api_server.py   ← API 服务启动入口
│   │   ├── serving_chat.py ← /v1/chat/completions 处理
│   │   └── serving_completion.py ← /v1/completions 处理
│   └── chat_utils.py       ← 对话模板处理工具
│
├── engine/                 ← 【引擎层】核心调度逻辑
│   ├── llm_engine.py       ← LLMEngine 同步引擎 ⭐ 核心文件
│   ├── async_llm_engine.py ← AsyncLLMEngine 异步引擎
│   └── arg_utils.py        ← 引擎参数定义
│
├── core/                   ← 【调度层】请求调度和显存管理
│   ├── scheduler.py        ← Scheduler 调度器 ⭐ 核心文件
│   ├── block_manager.py    ← Block Manager 块管理器 ⭐ 核心文件
│   └── block/              ← 块分配的具体实现
│       ├── cpu_gpu_block_allocator.py
│       └── prefix_caching_block.py  ← 前缀缓存的块分配
│
├── worker/                 ← 【执行层】GPU 上的计算
│   ├── worker.py           ← Worker 进程（管理GPU）
│   ├── model_runner.py     ← ModelRunner 模型执行器 ⭐ 核心文件
│   └── cache_engine.py     ← KV Cache 的 GPU 显存管理
│
├── model_executor/         ← 【模型层】模型定义
│   └── models/             ← 各种模型的实现
│       ├── llama.py        ← Llama 系列
│       ├── qwen2.py        ← Qwen2 系列
│       ├── chatglm.py      ← ChatGLM 系列
│       └── ...
│
├── attention/              ← 【注意力层】PagedAttention 实现
│   ├── backends/           
│   │   ├── flash_attn.py   ← FlashAttention 后端
│   │   └── xformers.py     ← xFormers 后端
│   └── ops/
│       └── paged_attn.py   ← PagedAttention 算子 ⭐ 核心文件
│
├── sampling_params.py      ← SamplingParams 定义
├── sequence.py             ← Sequence 数据结构（请求的内部表示）
├── config.py               ← 所有配置类
└── lora/                   ← LoRA 相关实现
    ├── lora_manager.py
    └── layers.py
""")


# =============================================
# 2. 核心调用链（一次推理请求的完整旅程）
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║          一次推理请求的完整旅程                                  ║
╚══════════════════════════════════════════════════════════════════╝

用户代码: llm.generate(["你好"], sampling_params)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] entrypoints/llm.py :: LLM.generate()                       │
│     - 接收用户输入                                               │
│     - 调用 tokenizer 把文本转成 token_ids                        │
│     - 创建 SequenceGroup（请求的内部表示）                        │
│     - 调用 engine.add_request() 提交请求                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ [2] engine/llm_engine.py :: LLMEngine.add_request()             │
│     - 请求进入 waiting 队列                                      │
│     - 开始主循环: while 有未完成请求:                              │
│         engine.step()                                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ [3] engine/llm_engine.py :: LLMEngine.step()                    │
│     这是每一步推理的核心流程：                                     │
│                                                                  │
│     3a. scheduler.schedule()                                     │
│         → Scheduler 决定这一步处理哪些请求                        │
│         → 区分 prefill（新请求）和 decode（继续生成）               │
│         → Block Manager 分配/回收物理块                           │
│                                                                  │
│     3b. worker.execute_model()                                   │
│         → 把选中的请求打包，送入 GPU 执行                          │
│         → ModelRunner 执行前向传播                                │
│         → PagedAttention kernel 处理分页的 KV Cache               │
│                                                                  │
│     3c. 处理输出                                                  │
│         → 采样得到新 token                                        │
│         → 检查是否满足停止条件                                     │
│         → 完成的请求移出调度，释放物理块                            │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] 返回结果                                                     │
│     - token_ids → tokenizer.decode() → 文本                     │
│     - 封装成 RequestOutput 返回给用户                             │
└─────────────────────────────────────────────────────────────────┘
""")


# =============================================
# 3. Scheduler 调度器详解
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              Scheduler 调度器详解                                ║
╚══════════════════════════════════════════════════════════════════╝

Scheduler 管理三个队列：

  waiting  ──→  running  ──→  完成
                  │
                  ↓ (显存不够时)
              swapped (换出到CPU)

调度策略：
1. 优先处理 running 队列中的请求（它们已经在生成中）
2. 如果显存够，从 waiting 队列取新请求进入 running
3. 如果显存不够，可能把 running 中优先级低的请求 swap 到 CPU
4. 当显存空闲时，从 swapped 队列恢复请求

关键源码位置：
  vllm/core/scheduler.py :: Scheduler._schedule()

调度器做的核心决策：
  - 这一步（iteration）处理哪些请求？
  - 哪些是 prefill（第一次处理，需要计算整个 prompt 的 KV）？
  - 哪些是 decode（已经在生成中，只需计算新 token）？
  - 显存够不够？需不需要抢占（preempt）某些请求？
""")


# =============================================
# 4. Block Manager 块管理器详解
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              Block Manager 块管理器详解                          ║
╚══════════════════════════════════════════════════════════════════╝

Block Manager 的职责：
  - 管理 GPU 和 CPU 上的物理块
  - 为请求分配和回收物理块
  - 实现 Copy-on-Write（写时复制）
  - 支持前缀缓存

核心数据结构：

  PhysicalBlock:
    ├── block_number: int     # 物理块编号
    ├── block_size: int       # 每块容纳多少 token
    └── ref_count: int        # 引用计数（用于 CoW）

  BlockTable（块表）:
    逻辑块号 → 物理块号 的映射
    类似操作系统的页表

工作流程示例：

  1. 新请求到来（6个token，block_size=4）
     → 分配 2 个物理块: [0→块7] [1→块3]
     → 块7: [t0, t1, t2, t3]  块3: [t4, t5, _, _]
  
  2. 生成新token t6
     → 块3还有空位 → 直接放入: 块3: [t4, t5, t6, _]
  
  3. 生成新token t7
     → 块3满了 → 分配新块: [2→块11]
     → 块11: [t7, _, _, _]
  
  4. 请求完成
     → 释放块7、块3、块11 → 加入空闲列表

关键源码位置：
  vllm/core/block_manager.py
  vllm/core/block/
""")


# =============================================
# 5. ModelRunner 模型执行器
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              ModelRunner 模型执行器                              ║
╚══════════════════════════════════════════════════════════════════╝

ModelRunner 负责在 GPU 上实际执行模型推理。

核心方法: execute_model()

工作流程：
  1. prepare_model_input()
     - 把多个请求的 token 拼成一个大 batch
     - 构建 attention metadata（告诉 PagedAttention 怎么读块）
     - 区分 prefill tokens 和 decode tokens
  
  2. model.forward()
     - 执行模型前向传播
     - 在 Attention 层使用 PagedAttention kernel
     - 得到 logits（每个 token 的概率分布）
  
  3. sample()
     - 根据 SamplingParams 采样
     - temperature → top_p → top_k → 采样
     - 得到新 token

性能优化技术：
  - CUDA Graph: 把多次 GPU kernel 调用"录制"成一个图，减少启动开销
  - FlashAttention: 优化的注意力计算 kernel，减少显存访问
  - Continuous Batching: prefill 和 decode 请求混合在同一个 batch 中

关键源码位置：
  vllm/worker/model_runner.py
""")


# =============================================
# 6. 推荐的源码阅读顺序
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              推荐的源码阅读顺序                                  ║
╚══════════════════════════════════════════════════════════════════╝

第一轮：理解接口和数据流（2-3天）
  1. vllm/entrypoints/llm.py        ← LLM 类，用户接口
  2. vllm/sampling_params.py        ← 采样参数定义
  3. vllm/sequence.py               ← Sequence 数据结构
  4. vllm/engine/llm_engine.py      ← LLMEngine，理解 step() 主循环

第二轮：理解调度和显存管理（3-5天）
  5. vllm/core/scheduler.py         ← Scheduler 调度逻辑
  6. vllm/core/block_manager.py     ← Block Manager 块管理
  7. vllm/core/block/               ← 块分配的具体实现

第三轮：理解 GPU 执行（3-5天）
  8. vllm/worker/worker.py          ← Worker 进程
  9. vllm/worker/model_runner.py    ← ModelRunner 执行器
  10. vllm/attention/               ← PagedAttention 实现

第四轮：理解模型层（按需）
  11. vllm/model_executor/models/   ← 选一个你熟悉的模型看
  12. vllm/lora/                    ← LoRA 实现

阅读技巧：
  ✓ 从 LLM.generate() 开始，逐层深入
  ✓ 重点理解数据结构（Sequence, SequenceGroup, Block）
  ✓ 在关键位置加 print/breakpoint 跟踪数据流
  ✓ 先理解正常路径，再看异常处理和优化
  ✗ 不要试图一次读完所有代码
  ✗ 不要一开始就看 CUDA kernel
""")


# =============================================
# 7. 调试技巧
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              vLLM 调试技巧                                      ║
╚══════════════════════════════════════════════════════════════════╝

1. 开启详细日志:
   VLLM_LOGGING_LEVEL=DEBUG python your_script.py

2. 关闭 CUDA Graph（错误信息更清晰）:
   llm = LLM(model="...", enforce_eager=True)

3. 查看调度器状态:
   在 scheduler.py 的 _schedule() 中加 print:
   print(f"waiting: {len(self.waiting)}, "
         f"running: {len(self.running)}, "
         f"swapped: {len(self.swapped)}")

4. 查看显存使用:
   import torch
   print(f"GPU显存: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
         f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB")

5. 使用 py-spy 分析性能瓶颈:
   pip install py-spy
   py-spy record -o profile.svg -- python your_script.py

6. 使用环境变量控制行为:
   VLLM_ATTENTION_BACKEND=FLASH_ATTN   # 指定注意力后端
   CUDA_VISIBLE_DEVICES=0,1             # 指定使用哪些 GPU
   VLLM_USE_MODELSCOPE=True             # 使用 ModelScope 下载模型
""")


# =============================================
# 总结：完整学习路线回顾
# =============================================

print("""
╔══════════════════════════════════════════════════════════════════╗
║              vLLM 学习路线 - 完整回顾                            ║
╚══════════════════════════════════════════════════════════════════╝

文件              │ 阶段            │ 内容
──────────────────┼─────────────────┼──────────────────────────────
vllm_learn1.py    │ 第一阶段:前置知识 │ LLM推理、KV Cache、瓶颈分析
vllm_learn2.py    │ 第二阶段:核心原理 │ PagedAttention、Continuous Batching
vllm_learn3.py    │ 第三阶段:基础实践 │ 安装、推理、Chat、API、调参
vllm_learn4.py    │ 第四阶段:进阶使用 │ 多卡、LoRA、量化、结构化输出
vllm_learn5.py    │ 第五阶段:源码解读 │ 架构、调用链、调度器、调试

建议学习顺序：
  第1-2阶段 → 先运行代码看效果，建立直觉
  第3阶段   → 动手实践，每个练习都跑一遍
  第4阶段   → 按需学习，用到哪个学哪个
  第5阶段   → 有余力再深入，不强求

祝学习顺利！🚀
""")
