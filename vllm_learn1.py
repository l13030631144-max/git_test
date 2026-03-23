"""
=============================================================
vLLM 学习路线 - 第一阶段：前置知识
=============================================================

在学习 vLLM 之前，你需要理解 3 个关键概念：
1. LLM 推理的基本流程
2. KV Cache 是什么
3. 为什么推理是瓶颈

学完这一部分，你会明白"vLLM 到底在解决什么问题"。
"""

# =============================================
# 1. 什么是 LLM 推理？
# =============================================
# 
# 简单说：你给模型一段话（prompt），模型一个字一个字地"续写"出回答。
# 这个"续写"过程就是推理（Inference）。
#
# 比如：
#   输入: "中国的首都是"
#   模型推理过程: "北" -> "京" -> "。"
#
# 注意：模型不是一次性输出所有文字，而是一个 token 一个 token 地生成。
# 每生成一个新 token，都要重新"看一遍"之前所有的内容。
#
# 这就引出了一个问题：每次都重新算之前的内容，太慢了！
# 解决办法就是 —— KV Cache。


# =============================================
# 2. KV Cache 是什么？（通俗版）
# =============================================
#
# 在 Transformer 的注意力机制（Attention）中，每个 token 会产生三个向量：
#   - Q（Query，查询）：我在找什么？
#   - K（Key，键）：我是什么？
#   - V（Value，值）：我包含什么信息？
#
# 【类比理解】想象你在图书馆找书：
#   - Q 就是你的"搜索关键词"
#   - K 就是每本书的"标签"
#   - V 就是每本书的"实际内容"
#
# 当模型生成第 N 个 token 时，需要用第 N 个 token 的 Q 
# 去和之前所有 token（1 到 N-1）的 K 做匹配，找到相关的 V。
#
# 问题来了：第 1 到 N-1 个 token 的 K 和 V，在之前的步骤已经算过了！
# 没必要重新算，直接缓存起来复用就好。
#
# 这个缓存就叫 KV Cache。
#
# 【KV Cache 的大小估算】
# 对于一个 7B 参数的模型（如 Llama-2-7B）：
#   - 32 层 × 32 个注意力头 × 128 维度 × 2（K和V）× 2字节（FP16）
#   - 每个 token 的 KV Cache ≈ 512 KB
#   - 如果序列长度 2048，一个请求的 KV Cache ≈ 1 GB！
#   - 同时处理 10 个请求，光 KV Cache 就要 10 GB 显存

def estimate_kv_cache_size(
    num_layers: int = 32,         # 模型层数
    num_heads: int = 32,          # 注意力头数
    head_dim: int = 128,          # 每个头的维度
    seq_length: int = 2048,       # 序列长度
    batch_size: int = 1,          # 批大小
    dtype_bytes: int = 2,         # 数据类型字节数（FP16=2）
) -> dict:
    """
    估算 KV Cache 的显存占用。
    
    这个函数帮你直观理解 KV Cache 有多大。
    """
    # 每个 token 在一层中的 KV Cache 大小
    # K 和 V 各一份，所以乘以 2
    per_token_per_layer = num_heads * head_dim * 2 * dtype_bytes  # 字节
    
    # 每个 token 在所有层的 KV Cache 大小
    per_token_all_layers = per_token_per_layer * num_layers
    
    # 一个请求的 KV Cache 大小
    per_request = per_token_all_layers * seq_length
    
    # 整个 batch 的 KV Cache 大小
    total = per_request * batch_size
    
    result = {
        "每个token每层的KV Cache": f"{per_token_per_layer / 1024:.1f} KB",
        "每个token所有层的KV Cache": f"{per_token_all_layers / 1024:.1f} KB",
        "单个请求的KV Cache": f"{per_request / (1024**3):.2f} GB",
        "整个batch的KV Cache": f"{total / (1024**3):.2f} GB",
    }
    return result


# 运行看看 Llama-2-7B 的 KV Cache 大小
print("=" * 50)
print("Llama-2-7B 的 KV Cache 大小估算")
print("=" * 50)
result = estimate_kv_cache_size(
    num_layers=32, num_heads=32, head_dim=128,
    seq_length=2048, batch_size=10
)
for k, v in result.items():
    print(f"  {k}: {v}")


# =============================================
# 3. 为什么推理是瓶颈？
# =============================================
#
# 【瓶颈一：显存浪费严重】
# 传统方法（如 HuggingFace）为每个请求预分配一段连续的显存空间给 KV Cache。
# 但问题是：
#   - 你不知道模型会生成多长的回答
#   - 所以只能按最大长度预分配（比如 2048 个 token）
#   - 但实际可能只用了 200 个 token
#   - 剩下 1848 个 token 的空间就浪费了！
#
# 这就像去餐厅，不管你吃多少，都给你摆一桌满汉全席的餐具。
#
# 【瓶颈二：批处理效率低】
# 传统的静态批处理（Static Batching）：
#   - 把多个请求凑成一批，一起处理
#   - 但必须等这一批全部完成，才能处理下一批
#   - 如果一个请求生成 10 个 token，另一个生成 1000 个 token
#   - 短的那个早就完了，但还得等长的那个
#   - GPU 在"等待"的时候就是浪费的
#
# 这就像坐大巴，要等所有人都到了才发车。
#
# 【瓶颈三：显存碎片化】
# 不同请求的 KV Cache 长度不同，频繁分配和释放会产生大量显存碎片。
# 就像停车场里大车小车乱停，明明总空间够，但就是停不下新的车。
#
# vLLM 的出现，就是为了解决这三个瓶颈！

print("\n" + "=" * 50)
print("传统方法 vs vLLM 的对比")
print("=" * 50)

comparison = {
    "显存管理": {
        "传统方法": "预分配连续大块显存，浪费严重（利用率约 20%~40%）",
        "vLLM":    "PagedAttention，按需分配小块，利用率接近 100%",
    },
    "批处理方式": {
        "传统方法": "静态批处理，等最长的请求完成才能处理下一批",
        "vLLM":    "连续批处理，请求随到随走，GPU 利用率大幅提升",
    },
    "吞吐量": {
        "传统方法": "受限于显存浪费和批处理效率",
        "vLLM":    "比 HuggingFace 快 2~24 倍（取决于场景）",
    },
}

for aspect, methods in comparison.items():
    print(f"\n  【{aspect}】")
    for method, desc in methods.items():
        print(f"    {method}: {desc}")


# =============================================
# 4. 一个简单的推理流程演示（伪代码）
# =============================================
#
# 下面用伪代码展示 LLM 自回归生成的过程，帮你理解推理的本质

def simple_autoregressive_generation(prompt_tokens: list, max_new_tokens: int = 10):
    """
    伪代码：展示 LLM 自回归生成的核心逻辑。
    
    注意：这不是真正的模型推理，只是帮助理解流程。
    """
    import random
    
    # 模拟的词汇表
    vocab = ["我", "是", "一个", "AI", "助手", "你好", "很高兴", "认识你", "。", "！"]
    
    generated_tokens = []
    kv_cache = {}  # 模拟 KV Cache
    
    print("\n自回归生成过程演示：")
    print(f"  输入 tokens: {prompt_tokens}")
    
    # ---- Prefill 阶段（预填充）----
    # 一次性处理所有输入 token，生成它们的 KV Cache
    print("\n  [Prefill 阶段] 处理所有输入 token...")
    for i, token in enumerate(prompt_tokens):
        # 计算这个 token 的 K 和 V，存入缓存
        kv_cache[i] = {"K": f"key_{token}", "V": f"val_{token}"}
    print(f"  KV Cache 已缓存 {len(kv_cache)} 个 token")
    
    # ---- Decode 阶段（逐步解码）----
    # 一个 token 一个 token 地生成
    print("\n  [Decode 阶段] 逐个生成新 token...")
    for step in range(max_new_tokens):
        # 新 token 的 Q 与所有已有的 K 做注意力计算
        # （这里用随机模拟）
        new_token = random.choice(vocab)
        
        # 把新 token 的 KV 也加入缓存
        pos = len(prompt_tokens) + step
        kv_cache[pos] = {"K": f"key_{new_token}", "V": f"val_{new_token}"}
        
        generated_tokens.append(new_token)
        print(f"    Step {step + 1}: 生成 '{new_token}' (KV Cache 大小: {len(kv_cache)})")
        
        # 遇到句号就停止
        if new_token in ["。", "！"]:
            break
    
    print(f"\n  最终生成: {''.join(generated_tokens)}")
    return generated_tokens


print("\n" + "=" * 50)
print("LLM 自回归生成流程演示")
print("=" * 50)
simple_autoregressive_generation(
    prompt_tokens=["你", "好", "，", "请", "介绍"],
    max_new_tokens=8
)


print("\n" + "=" * 50)
print("第一阶段学习完成！")
print("=" * 50)
print("""
关键知识点回顾：
1. LLM 推理是自回归的：一个 token 一个 token 地生成
2. KV Cache 缓存了之前 token 的 Key 和 Value，避免重复计算
3. KV Cache 非常占显存（7B 模型，10 个请求就要 ~10GB）
4. 传统方法有三大瓶颈：显存浪费、批处理效率低、显存碎片化
5. vLLM 就是为解决这些瓶颈而生的

下一步 → 运行 vllm_learn2.py 学习 vLLM 的核心原理（PagedAttention）
""")
