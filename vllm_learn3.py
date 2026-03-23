"""
=============================================================
vLLM 学习路线 - 第三阶段：基础实践（动手写代码！）
=============================================================

这一阶段你将真正上手 vLLM，从安装到完成第一次推理。
包含 6 个练习，由简到难。

前提：你需要有 GPU 环境（至少一张显卡，推荐 >= 16GB 显存）。
如果显存较小，可以选用小模型如 Qwen/Qwen2.5-0.5B-Instruct。
"""

# =============================================
# 练习 0：安装 vLLM
# =============================================
#
# 在终端中运行：
#
#   pip install vllm
#
# 验证安装：
#   python -c "import vllm; print(vllm.__version__)"
#
# 注意事项：
#   - vLLM 需要 CUDA 环境（NVIDIA GPU）
#   - 推荐 Python 3.9 ~ 3.12
#   - 如果安装慢，可以用国内镜像：
#     pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
#
# 如果你没有 GPU，可以先阅读代码，理解用法，后续找 GPU 环境再运行。


# =============================================
# 练习 1：最简单的离线推理（Offline Inference）
# =============================================
# 这是 vLLM 最基础的用法：加载模型 → 输入文本 → 得到结果

def exercise_1_basic_inference():
    """
    最基础的 vLLM 推理示例。
    
    核心概念：
    - LLM: vLLM 的主类，负责加载模型
    - SamplingParams: 控制生成行为的参数
    - generate(): 执行推理
    """
    from vllm import LLM, SamplingParams
    
    # ---- 第一步：设置采样参数 ----
    # 这些参数控制模型"怎么生成"
    sampling_params = SamplingParams(
        temperature=0.8,    # 温度：越高越随机，越低越确定
                            # 0 = 总是选概率最高的（贪心）
                            # 1 = 按原始概率采样
                            # >1 = 更随机
        top_p=0.95,         # 核采样：只从累计概率前 95% 的 token 中采样
                            # 过滤掉低概率的"噪声" token
        max_tokens=256,     # 最多生成多少个 token
        logprobs=1,         # 返回每个 token 的对数概率（开启后才有 cumulative_logprob）
    )
    
    # ---- 第二步：加载模型 ----
    # vLLM 会自动下载模型（第一次会比较慢）
    # 你也可以用本地路径，比如: model="/path/to/your/model"
    llm = LLM(
        model="./Qwen2.5-0.5B-Instruct",  # 本地模型路径
        gpu_memory_utilization=0.6,  # 降低显存利用率，适配 6GB 显卡
        max_model_len=2048,          # 限制上下文长度，节省显存
        enforce_eager=True,          # 关闭 CUDA Graph，调试时更友好
        seed=42,                    # 固定随机种子，结果可复现
    )
    
    # ---- 第三步：准备输入 ----
    prompts = [
        "请用一句话解释什么是人工智能。",
        "Python 和 Java 的主要区别是什么？",
        "写一首关于春天的五言绝句。",
    ]
    
    # ---- 第四步：执行推理 ----
    outputs = llm.generate(prompts, sampling_params)
    
    # ---- 第五步：解析输出 ----
    print("=" * 60)
    print("练习 1：基础离线推理结果")
    print("=" * 60)
    
    for output in outputs:
        prompt = output.prompt           # 输入文本
        generated = output.outputs[0]    # 生成结果（可能有多个，取第一个）
        
        print(f"\n输入: {prompt}")
        print(f"输出: {generated.text}")
        print(f"生成 token 数: {len(generated.token_ids)}")
        # cumulative_logprob 是所有 token 的对数概率之和，可以衡量模型的"信心"
        if generated.cumulative_logprob is not None:
            print(f"累计对数概率: {generated.cumulative_logprob:.2f}")
    
    return outputs


# =============================================
# 练习 2：理解 SamplingParams（采样参数详解）
# =============================================

def exercise_2_sampling_params():
    """
    对比不同采样参数的效果。
    
    关键参数一览：
    - temperature: 控制随机性
    - top_p: 核采样
    - top_k: 只从概率最高的 K 个 token 中采样
    - n: 对每个 prompt 生成 n 个不同结果
    - best_of: 生成 best_of 个结果，返回最好的 n 个
    - presence_penalty: 鼓励模型谈论新话题
    - frequency_penalty: 减少重复用词
    - stop: 遇到指定字符串时停止生成
    """
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="./Qwen2.5-0.5B-Instruct",
              gpu_memory_utilization=0.7, max_model_len=2048)
    prompt = "请给我讲一个笑话："
    
    # 对比不同 temperature
    print("=" * 60)
    print("练习 2：不同 temperature 的效果对比")
    print("=" * 60)
    
    for temp in [0.0, 0.5, 1.0, 1.5]:
        params = SamplingParams(
            temperature=temp,
            max_tokens=100,
            # temperature=0 时必须关闭 top_p（或设为1）
            top_p=1.0 if temp == 0 else 0.95,
        )
        output = llm.generate([prompt], params)[0]
        print(f"\ntemperature={temp}:")
        print(f"  {output.outputs[0].text[:100]}...")  # 只显示前100字
    
    # 生成多个结果
    print("\n" + "=" * 60)
    print("生成多个不同回答 (n=3)")
    print("=" * 60)
    
    params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        n=3,            # 生成 3 个不同的结果
    )
    output = llm.generate([prompt], params)[0]
    for i, result in enumerate(output.outputs):
        print(f"\n  回答{i+1}: {result.text[:80]}...")
    
    # 使用 stop 参数
    print("\n" + "=" * 60)
    print("使用 stop 参数控制停止条件")
    print("=" * 60)
    
    params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
        stop=["。", "！"],  # 遇到句号或感叹号就停止
    )
    output = llm.generate(["请介绍一下北京"], params)[0]
    print(f"  输出（遇到句号/感叹号即停止）: {output.outputs[0].text}")


# =============================================
# 练习 3：Chat 对话模式
# =============================================

def exercise_3_chat():
    """
    使用 vLLM 的 chat 接口进行多轮对话。
    
    chat() 接口会自动应用模型的对话模板（chat template），
    你不需要手动拼接 <|im_start|> 这样的特殊标记。
    """
    from vllm import LLM, SamplingParams
    
    llm = LLM(model="./Qwen2.5-0.5B-Instruct",
              gpu_memory_utilization=0.7, max_model_len=2048)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
    
    print("=" * 60)
    print("练习 3：Chat 对话模式")
    print("=" * 60)
    
    # ---- 单轮对话 ----
    messages_list = [
        # 每个元素是一个完整的对话（消息列表）
        [
            {"role": "system", "content": "你是一个友好的AI助手。"},
            {"role": "user", "content": "什么是机器学习？请用简单的话解释。"},
        ],
        [
            {"role": "system", "content": "你是一个Python专家。"},
            {"role": "user", "content": "列表推导式和 map 函数哪个更好？"},
        ],
    ]
    
    # chat() 接口直接接受消息列表
    outputs = llm.chat(messages_list, sampling_params)
    
    for i, output in enumerate(outputs):
        print(f"\n对话 {i+1}:")
        print(f"  用户: {messages_list[i][-1]['content']}")
        print(f"  AI: {output.outputs[0].text}")
    
    # ---- 多轮对话 ----
    print("\n" + "-" * 40)
    print("多轮对话示例:")
    print("-" * 40)
    
    # 第一轮
    conversation = [
        {"role": "system", "content": "你是一个数学老师，用通俗易懂的方式讲解。"},
        {"role": "user", "content": "什么是微积分？"},
    ]
    
    output1 = llm.chat([conversation], sampling_params)[0]
    ai_reply1 = output1.outputs[0].text
    print(f"\n  用户: 什么是微积分？")
    print(f"  AI: {ai_reply1[:150]}...")
    
    # 第二轮：把上一轮的回答加入对话历史
    conversation.append({"role": "assistant", "content": ai_reply1})
    conversation.append({"role": "user", "content": "能举个生活中的例子吗？"})
    
    output2 = llm.chat([conversation], sampling_params)[0]
    ai_reply2 = output2.outputs[0].text
    print(f"\n  用户: 能举个生活中的例子吗？")
    print(f"  AI: {ai_reply2[:150]}...")


# =============================================
# 练习 4：启动 OpenAI 兼容的 API 服务
# =============================================

def exercise_4_api_server():
    """
    这个练习不是直接运行的 Python 函数，而是在终端中操作。
    
    vLLM 提供了兼容 OpenAI API 格式的服务，这意味着你可以
    用任何 OpenAI SDK 的客户端直接对接 vLLM！
    """
    print("=" * 60)
    print("练习 4：启动 OpenAI 兼容 API 服务")
    print("=" * 60)
    
    print("""
【步骤 1】在终端中启动 vLLM 服务:

    python -m vllm.entrypoints.openai.api_server \\
        --model ./Qwen2.5-0.5B-Instruct \\
        --host 0.0.0.0 \\
        --port 8000

    常用参数：
        --tensor-parallel-size 2     # 多卡并行（2张卡）
        --gpu-memory-utilization 0.8 # 显存利用率
        --max-model-len 4096         # 最大上下文长度
        --dtype auto                 # 数据类型（auto/half/float16/bfloat16）


【步骤 2】用 curl 测试:

    # 补全（Completions）
    curl http://localhost:8000/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "./Qwen2.5-0.5B-Instruct",
            "prompt": "你好，请介绍一下你自己",
            "max_tokens": 100,
            "temperature": 0.7
        }'
    
    # 对话（Chat Completions）
    curl http://localhost:8000/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "./Qwen2.5-0.5B-Instruct",
            "messages": [
                {"role": "user", "content": "什么是深度学习？"}
            ],
            "max_tokens": 200
        }'


【步骤 3】用 Python OpenAI SDK 调用:
    （见下方代码）
""")


def exercise_4_client_code():
    """
    用 OpenAI SDK 调用 vLLM 服务。
    
    先确保 vLLM API 服务已经启动（见 exercise_4_api_server 的说明）。
    安装 SDK: pip install openai
    """
    from openai import OpenAI
    
    # 关键：base_url 指向你的 vLLM 服务地址
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",  # vLLM 默认不需要 API key
    )
    
    # ---- Chat Completions（最常用）----
    print("Chat 模式:")
    response = client.chat.completions.create(
        model="./Qwen2.5-0.5B-Instruct",
        messages=[
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "用三句话介绍Python语言。"},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    print(f"  {response.choices[0].message.content}")
    
    # ---- 流式输出（Streaming）----
    print("\n流式输出:")
    stream = client.chat.completions.create(
        model="./Qwen2.5-0.5B-Instruct",
        messages=[
            {"role": "user", "content": "写一首关于月亮的诗"},
        ],
        temperature=0.8,
        max_tokens=200,
        stream=True,  # 开启流式！
    )
    
    print("  ", end="")
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()  # 换行
    
    # ---- 查看模型列表 ----
    print("\n可用模型:")
    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")

# exercise_4_client_code()
# print("\n提示：上面的代码需要在 vLLM API 服务启动后才能运行。请先按照 exercise_4_api_server 的说明操作。")


# =============================================
# 练习 5：批量推理与性能测试
# =============================================

def exercise_5_benchmark():
    """
    测试 vLLM 的吞吐量，感受它的速度。
    """
    import time
    from vllm import LLM, SamplingParams
    
    print("=" * 60)
    print("练习 5：批量推理性能测试")
    print("=" * 60)
    
    llm = LLM(model="./Qwen2.5-0.5B-Instruct",
              gpu_memory_utilization=0.7, max_model_len=2048)

    # 准备大量请求
    prompts = [f"请用一句话描述数字{i}的特点。" for i in range(100)]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    # 计时
    print(f"\n开始批量推理 {len(prompts)} 个请求...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    elapsed = time.time() - start_time
    
    # 统计
    total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    
    print(f"\n性能统计:")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  请求数: {len(prompts)}")
    print(f"  总输入 tokens: {total_input_tokens}")
    print(f"  总输出 tokens: {total_output_tokens}")
    print(f"  输出吞吐量: {total_output_tokens / elapsed:.1f} tokens/s")
    print(f"  请求吞吐量: {len(prompts) / elapsed:.1f} requests/s")
    
    # 显示几个示例结果
    print(f"\n示例输出（前3个）:")
    for output in outputs[:3]:
        print(f"  输入: {output.prompt}")
        print(f"  输出: {output.outputs[0].text.strip()}")
        print()


# =============================================
# 练习 6：关键参数调优指南
# =============================================

def exercise_6_tuning_guide():
    """
    介绍 vLLM 的关键初始化参数和调优技巧。
    """
    print("=" * 60)
    print("练习 6：vLLM 关键参数调优指南")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                     LLM() 初始化关键参数                        │
├───────────────────────┬─────────────────────────────────────────┤
│ 参数                  │ 说明                                    │
├───────────────────────┼─────────────────────────────────────────┤
│ model                 │ 模型路径（HuggingFace名 或 本地路径）     │
│ tensor_parallel_size  │ 张量并行度（用几张卡），默认1             │
│ gpu_memory_utilization│ GPU显存利用率，默认0.9                    │
│                       │ 降低可预留显存给其他程序                  │
│ max_model_len         │ 最大上下文长度，默认读取模型配置           │
│                       │ 显存不够时可以调小                       │
│ dtype                 │ 数据类型: "auto"/"half"/"bfloat16"       │
│ quantization          │ 量化方式: "awq"/"gptq"/"fp8" 等          │
│ enforce_eager         │ True=关闭CUDA Graph，调试时有用           │
│ trust_remote_code     │ 是否信任模型仓库的自定义代码              │
│ seed                  │ 随机种子，用于结果复现                    │
│ swap_space            │ CPU swap空间（GB），默认4                 │
│ enable_prefix_caching │ 开启前缀缓存，相同前缀复用KV Cache        │
└───────────────────────┴─────────────────────────────────────────┘

常见调优场景：

1. 显存不够用？
   → 降低 gpu_memory_utilization（如 0.7）
   → 降低 max_model_len（如 2048）
   → 使用量化模型（quantization="awq"）

2. 想要更高吞吐？
   → 增大 gpu_memory_utilization（如 0.95）
   → 使用多卡（tensor_parallel_size=2）
   → 开启前缀缓存（enable_prefix_caching=True）

3. 调试或开发时？
   → enforce_eager=True（关闭CUDA Graph，错误信息更清晰）
   → seed=42（固定随机种子，结果可复现）
""")
    
    # 示例：不同配置的 LLM 初始化
    print("常用初始化配置示例：")
    print("""
# 基础配置（单卡）
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 显存较小的卡（如 16GB）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=2048,
)

# 多卡并行（2张卡）
llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    tensor_parallel_size=2,
)

# 量化模型（节省显存）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization="awq",
)

# 开启前缀缓存（适合大量相似请求）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True,
)

# 调试模式
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enforce_eager=True,
    seed=42,
)
""")


# =============================================
# 主程序：选择要运行的练习
# =============================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║     vLLM 第三阶段：基础实践练习          ║
╠══════════════════════════════════════════╣
║  1. 基础离线推理                         ║
║  2. 采样参数详解                         ║
║  3. Chat 对话模式                        ║
║  4. API 服务说明                         ║
║  5. 批量推理性能测试                     ║
║  6. 参数调优指南                         ║
║  0. 全部运行（需要 GPU + 模型）          ║
╚══════════════════════════════════════════╝
    """)
    
    choice = input("请选择练习编号 (1-6, 或 0 全部运行): ").strip()
    
    exercises = {
        "1": exercise_1_basic_inference,
        "2": exercise_2_sampling_params,
        "3": exercise_3_chat,
        "4": exercise_4_api_server,
        "5": exercise_5_benchmark,
        "6": exercise_6_tuning_guide,
    }
    
    if choice == "0":
        for name, func in exercises.items():
            print(f"\n{'#' * 60}")
            print(f"# 运行练习 {name}")
            print(f"{'#' * 60}")
            func()
    elif choice in exercises:
        exercises[choice]()
    else:
        print("无效选择，请输入 0-6")
        # 默认展示参数调优指南（不需要GPU也能看）
        exercise_6_tuning_guide()
