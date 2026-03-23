"""
=============================================================
vLLM 学习路线 - 第二阶段：核心原理
=============================================================

这一阶段学习 vLLM 的两大核心技术：
1. PagedAttention（分页注意力）—— 解决显存浪费和碎片化
2. Continuous Batching（连续批处理）—— 解决批处理效率低

学完你就能理解 vLLM 为什么快。
"""

# =============================================
# 1. PagedAttention —— vLLM 的灵魂
# =============================================
#
# 【问题回顾】
# 传统方法给每个请求分配一大块连续显存来存 KV Cache：
#
#   请求A: [████████████________________]  ← 预分配2048，实际用了800
#   请求B: [████████████████████________]  ← 预分配2048，实际用了1200  
#   请求C: [____________________________]  ← 想加入，但剩余空间不够连续分配
#
# 大量空间被浪费！而且有碎片！
#
# 【PagedAttention 的解决方案】
# 借鉴操作系统的虚拟内存管理思想：
#
# 操作系统怎么管理内存？
#   - 把物理内存分成固定大小的"页"（Page），通常 4KB
#   - 进程需要内存时，按需分配页，不需要连续
#   - 通过"页表"记录逻辑地址到物理地址的映射
#
# PagedAttention 怎么管理 KV Cache？
#   - 把 GPU 显存分成固定大小的"块"（Block），每块存几个 token 的 KV
#   - 请求需要 KV Cache 时，按需分配块，不需要连续
#   - 通过"块表"（Block Table）记录逻辑块到物理块的映射
#
# 效果：
#   物理显存块: [A₁][B₁][A₂][B₂][C₁][A₃][B₃][C₂]...
#   
#   请求A的块表: 逻辑块0→物理块0, 逻辑块1→物理块2, 逻辑块2→物理块5
#   请求B的块表: 逻辑块0→物理块1, 逻辑块1→物理块3, 逻辑块2→物理块6
#   请求C的块表: 逻辑块0→物理块4, 逻辑块1→物理块7
#
# 没有浪费！没有碎片！

def demonstrate_paged_attention():
    """
    用 Python 模拟 PagedAttention 的核心思想。
    
    这不是真实实现，而是帮你直观理解"分页管理KV Cache"的概念。
    """
    
    BLOCK_SIZE = 4        # 每个 block 能存 4 个 token 的 KV
    TOTAL_BLOCKS = 16     # GPU 显存总共划分成 16 个 block
    
    # ---- 物理块管理器 ----
    class BlockManager:
        """模拟 vLLM 的 Block Manager"""
        
        def __init__(self, total_blocks: int):
            # 空闲块列表（就像操作系统的空闲页框列表）
            self.free_blocks = list(range(total_blocks))
            # 每个物理块的内容（模拟）
            self.physical_blocks = [None] * total_blocks
            
        def allocate_block(self) -> int:
            """分配一个空闲物理块，返回块号"""
            if not self.free_blocks:
                raise MemoryError("显存不足！没有空闲块了")
            block_id = self.free_blocks.pop(0)
            return block_id
        
        def free_block(self, block_id: int):
            """释放一个物理块"""
            self.physical_blocks[block_id] = None
            self.free_blocks.append(block_id)
            
        def get_free_count(self) -> int:
            return len(self.free_blocks)
    
    # ---- 请求的 KV Cache 管理 ----
    class Request:
        """模拟一个推理请求"""
        
        def __init__(self, request_id: str, prompt_tokens: list):
            self.id = request_id
            self.tokens = prompt_tokens.copy()
            # 块表：逻辑块号 → 物理块号（类似操作系统的页表）
            self.block_table = {}
            # 当前最后一个块已使用的 slot 数
            self.last_block_usage = 0
            
        def get_logical_block_count(self) -> int:
            """需要多少个逻辑块"""
            import math
            return math.ceil(len(self.tokens) / BLOCK_SIZE)
    
    # ---- 开始演示 ----
    print("=" * 60)
    print("PagedAttention 工作原理演示")
    print("=" * 60)
    print(f"配置: 每块容纳 {BLOCK_SIZE} 个 token, 总共 {TOTAL_BLOCKS} 个物理块\n")
    
    manager = BlockManager(TOTAL_BLOCKS)
    requests = {}
    
    # 第一步：请求A到来，有 6 个 prompt token
    print("【Step 1】请求A到来，prompt = '今天天气真不错' (6个token)")
    req_a = Request("A", list("今天天气真不"))
    # 需要 ceil(6/4) = 2 个块
    for logical_idx in range(req_a.get_logical_block_count()):
        physical_id = manager.allocate_block()
        req_a.block_table[logical_idx] = physical_id
        # 填入 token 数据
        start = logical_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, len(req_a.tokens))
        manager.physical_blocks[physical_id] = req_a.tokens[start:end]
    
    req_a.last_block_usage = len(req_a.tokens) % BLOCK_SIZE or BLOCK_SIZE
    requests["A"] = req_a
    
    print(f"  分配了 {len(req_a.block_table)} 个块")
    print(f"  块表: {req_a.block_table}")
    print(f"  剩余空闲块: {manager.get_free_count()}\n")
    
    # 第二步：请求B到来，有 10 个 prompt token
    print("【Step 2】请求B到来，prompt = '请帮我写一首关于春天的诗歌' (10个token)")
    req_b = Request("B", list("请帮我写一首关于春天"))
    for logical_idx in range(req_b.get_logical_block_count()):
        physical_id = manager.allocate_block()
        req_b.block_table[logical_idx] = physical_id
        start = logical_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, len(req_b.tokens))
        manager.physical_blocks[physical_id] = req_b.tokens[start:end]
    
    req_b.last_block_usage = len(req_b.tokens) % BLOCK_SIZE or BLOCK_SIZE
    requests["B"] = req_b
    
    print(f"  分配了 {len(req_b.block_table)} 个块")
    print(f"  块表: {req_b.block_table}")
    print(f"  剩余空闲块: {manager.get_free_count()}\n")
    
    # 第三步：请求A生成新 token，需要扩展 KV Cache
    print("【Step 3】请求A生成新 token '错'")
    new_token = "错"
    req_a.tokens.append(new_token)
    
    # 检查最后一个块是否还有空间
    last_logical = max(req_a.block_table.keys())
    if req_a.last_block_usage < BLOCK_SIZE:
        # 最后一个块还有空位，直接放进去
        physical_id = req_a.block_table[last_logical]
        manager.physical_blocks[physical_id].append(new_token)
        req_a.last_block_usage += 1
        print(f"  最后一个块还有空位，直接追加（无需新分配）")
    else:
        # 最后一个块满了，需要新分配一个块
        new_logical = last_logical + 1
        physical_id = manager.allocate_block()
        req_a.block_table[new_logical] = physical_id
        manager.physical_blocks[physical_id] = [new_token]
        req_a.last_block_usage = 1
        print(f"  需要新块，分配物理块 {physical_id}")
    
    print(f"  块表: {req_a.block_table}")
    print(f"  剩余空闲块: {manager.get_free_count()}\n")
    
    # 第四步：请求A完成，释放所有块
    print("【Step 4】请求A完成，释放所有块")
    for logical_idx, physical_id in req_a.block_table.items():
        manager.free_block(physical_id)
    print(f"  释放了 {len(req_a.block_table)} 个块")
    print(f"  剩余空闲块: {manager.get_free_count()}\n")
    
    # 显示最终物理块状态
    print("【物理块最终状态】")
    for i, content in enumerate(manager.physical_blocks):
        status = f"{''.join(content)}" if content else "空闲"
        print(f"  块{i:2d}: [{status}]")
    
    return manager


demonstrate_paged_attention()


# =============================================
# 2. PagedAttention 的三大优势
# =============================================

print("\n" + "=" * 60)
print("PagedAttention 的三大优势")
print("=" * 60)

print("""
【优势1：几乎零浪费】
  传统方法: 预分配 2048 token 空间，实际用 500 → 浪费 75%
  PagedAttention: 用了 500 个 token → 分配 ceil(500/16)=32 个块
                  最多浪费最后一个块的几个 slot（不到一个块）

【优势2：无碎片化】
  传统方法: 需要连续空间，释放后会产生碎片
  PagedAttention: 块可以散布在显存任何位置，释放后立即可复用

【优势3：支持 KV Cache 共享（Copy-on-Write）】
  场景：Beam Search 或并行采样时，多个候选序列共享前缀
  传统方法: 每个候选序列复制一份完整的 KV Cache
  PagedAttention: 多个序列可以指向同一个物理块，修改时才复制
  
  例如: 请求要求生成3个不同回答（n=3）
    逻辑视图：
      候选1: [共享前缀块][共享前缀块][独立块1]
      候选2: [共享前缀块][共享前缀块][独立块2]  
      候选3: [共享前缀块][共享前缀块][独立块3]
    
    物理视图: 前缀只存一份！节省 2/3 的前缀空间！
""")


# =============================================
# 3. Continuous Batching（连续批处理）
# =============================================
#
# 【传统 Static Batching 的问题】
#
# 时间轴 →→→→→→→→→→→→→→→→→→→
# 请求A: [████████████]              ← 生成 12 个 token 后完成
# 请求B: [████████████████████████]  ← 生成 24 个 token 后完成
# 请求C: [████████]                  ← 生成 8 个 token 后完成
#
# Static Batching: A和C早就完了，GPU 还在等 B 完成
# A完成后到B完成之间，A占用的GPU资源在空转！
#
#
# 【Continuous Batching 的做法】
#
# 时间轴 →→→→→→→→→→→→→→→→→→→
# 请求A: [████████████]
# 请求C: [████████]
# 请求D:          [████████████████]  ← C完成后，D立即补上
# 请求B: [████████████████████████]
# 请求E:              [████████████]  ← A完成后，E立即补上
#
# GPU 始终满载运行！

def demonstrate_continuous_batching():
    """
    模拟 Static Batching vs Continuous Batching 的效率差异。
    """
    import random
    random.seed(42)
    
    print("\n" + "=" * 60)
    print("Static Batching vs Continuous Batching 对比演示")
    print("=" * 60)
    
    # 模拟 8 个请求，每个需要不同步数完成
    requests = [
        {"id": f"请求{i}", "steps_needed": random.randint(3, 15)}
        for i in range(8)
    ]
    
    print("\n所有请求及其需要的步数:")
    for req in requests:
        bar = "█" * req["steps_needed"]
        print(f"  {req['id']}: {bar} ({req['steps_needed']}步)")
    
    # ---- Static Batching ----
    print("\n--- Static Batching (批大小=4) ---")
    batch_size = 4
    total_steps_static = 0
    
    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start:batch_start + batch_size]
        # 必须等最长的请求完成
        max_steps = max(r["steps_needed"] for r in batch)
        batch_names = [r["id"] for r in batch]
        
        # 计算浪费的步数
        total_compute = max_steps * len(batch)
        useful_compute = sum(r["steps_needed"] for r in batch)
        wasted = total_compute - useful_compute
        
        print(f"  批次 {batch_names}:")
        print(f"    等待最长请求: {max_steps}步")
        print(f"    总计算量: {total_compute}, 有效: {useful_compute}, 浪费: {wasted}")
        total_steps_static += max_steps
    
    print(f"  总耗时: {total_steps_static} 步\n")
    
    # ---- Continuous Batching ----
    print("--- Continuous Batching (最大并发=4) ---")
    max_concurrent = 4
    pending = list(range(len(requests)))  # 等待的请求索引
    active = {}  # 正在处理: {请求索引: 剩余步数}
    total_steps_continuous = 0
    completed = []
    
    while pending or active:
        # 填满并发槽位
        while pending and len(active) < max_concurrent:
            idx = pending.pop(0)
            active[idx] = requests[idx]["steps_needed"]
        
        # 所有活跃请求推进 1 步
        total_steps_continuous += 1
        finished_this_step = []
        
        for idx in list(active.keys()):
            active[idx] -= 1
            if active[idx] == 0:
                finished_this_step.append(idx)
        
        # 移除完成的请求（腾出槽位）
        for idx in finished_this_step:
            del active[idx]
            completed.append(requests[idx]["id"])
            # 下一步循环会立即填入新请求！
    
    print(f"  完成顺序: {' → '.join(completed)}")
    print(f"  总耗时: {total_steps_continuous} 步\n")
    
    speedup = total_steps_static / total_steps_continuous
    print(f"  速度提升: {speedup:.1f}x")
    print(f"  (Continuous Batching 让 GPU 始终满载，不浪费等待时间)")


demonstrate_continuous_batching()


# =============================================
# 4. vLLM 整体架构概览
# =============================================

print("\n" + "=" * 60)
print("vLLM 整体架构概览")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────┐
│                    用户请求                          │
│            (API / Python 调用)                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  LLM Engine                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Tokenizer  │  │  Scheduler   │  │   Block    │ │
│  │  (分词器)    │  │  (调度器)     │  │  Manager   │ │
│  │             │  │              │  │ (块管理器)  │ │
│  │ 文本→Token  │  │ 决定处理哪些  │  │ 管理物理块  │ │
│  │ Token→文本  │  │ 请求，按什么  │  │ 的分配和    │ │
│  │             │  │ 顺序处理     │  │ 释放       │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    Worker                            │
│  ┌─────────────────────────────────────────────┐    │
│  │              Model Runner                    │    │
│  │  ┌───────────┐  ┌──────────────────────┐   │    │
│  │  │   Model   │  │  PagedAttention      │   │    │
│  │  │  (模型)    │  │  (分页注意力 Kernel) │   │    │
│  │  │           │  │                      │   │    │
│  │  │ Llama /   │  │  高效的 CUDA Kernel  │   │    │
│  │  │ Qwen /    │  │  直接操作分页的      │   │    │
│  │  │ ChatGLM   │  │  KV Cache           │   │    │
│  │  └───────────┘  └──────────────────────┘   │    │
│  └─────────────────────────────────────────────┘    │
│                       GPU                            │
└─────────────────────────────────────────────────────┘

核心流程：
  1. 用户发来请求 → Tokenizer 把文本转成 token 序列
  2. Scheduler 决定这一步处理哪些请求（Continuous Batching）
  3. Block Manager 为需要的请求分配/回收显存块（PagedAttention）
  4. Worker 把这一批请求送入 GPU，Model Runner 执行推理
  5. 得到新 token → 判断是否结束 → 没结束就回到步骤 2
  6. 结束的请求 → Tokenizer 把 token 转回文本 → 返回给用户
""")


print("\n" + "=" * 60)
print("第二阶段学习完成！")
print("=" * 60)
print("""
关键知识点回顾：
1. PagedAttention 把 KV Cache 分成固定大小的块，按需分配
   → 类比操作系统的虚拟内存分页机制
2. 块表（Block Table）记录逻辑块到物理块的映射
   → 类比操作系统的页表
3. 支持 Copy-on-Write，多个序列可共享前缀的 KV Cache
4. Continuous Batching 让请求随到随走
   → 类比"随到随上"的公交车 vs "等人齐发车"的大巴
5. vLLM 架构: Engine → Scheduler + Block Manager → Worker → GPU

下一步 → 运行 vllm_learn3.py 开始动手实践！
""")
