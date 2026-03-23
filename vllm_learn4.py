"""
=============================================================
vLLM 学习路线 - 第四阶段：进阶使用
=============================================================

这一阶段覆盖 vLLM 的高级功能：
1. 多卡推理（Tensor Parallel）
2. 流式输出（Streaming）
3. LoRA 适配器动态加载
4. 量化模型推理
5. 前缀缓存（Prefix Caching）
6. 结构化输出（Guided Decoding）
7. 多模态模型（Vision Language Model）
"""

# =============================================
# 1. 多卡推理（Tensor Parallel）
# =============================================
#
# 当模型太大，一张卡放不下时，用多张卡来分担。
#
# 【原理】
# Tensor Parallelism（张量并行）：把模型的每一层的权重矩阵切开，
# 分到多张卡上。每张卡负责计算一部分，然后合并结果。
#
#   单卡:  GPU0 负责整个矩阵 [████████████████]
#   2卡:   GPU0 [████████]  GPU1 [████████]
#   4卡:   GPU0 [████] GPU1 [████] GPU2 [████] GPU3 [████]
#
# 注意：tensor_parallel_size 必须能整除模型的注意力头数。

def exercise_tensor_parallel():
    """
    多卡推理示例。
    需要多张 GPU。
    """
    from vllm import LLM, SamplingParams
    
    # 使用 2 张卡加载大模型
    llm = LLM(
        model="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=2,     # 使用 2 张 GPU
        # pipeline_parallel_size=2, # 流水线并行（另一种多卡策略，较少用）
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    outputs = llm.generate(["请详细解释量子计算的原理。"], sampling_params)
    print(outputs[0].outputs[0].text)


# =============================================
# 2. 流式输出（Streaming）—— 异步引擎
# =============================================
#
# 用户体验更好：一个字一个字地蹦出来，而不是等全部生成完。
# vLLM 通过 AsyncLLMEngine 实现异步流式输出。

def exercise_streaming_offline():
    """
    离线模式下的流式输出（不需要启动服务）。
    
    注意：vLLM 的离线 LLM 类不直接支持 streaming，
    但可以通过 AsyncLLMEngine 实现。
    实际工程中，流式输出主要通过 API 服务实现。
    """
    print("=" * 60)
    print("流式输出 - 通过 API 服务实现")
    print("=" * 60)
    
    print("""
方式一：通过 OpenAI SDK (推荐)
    先启动 vLLM API 服务，然后：
    
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
    
    stream = client.chat.completions.create(
        model="./Qwen2.5-0.5B-Instruct",
        messages=[{"role": "user", "content": "讲个故事"}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

方式二：通过 AsyncLLMEngine (底层方式)
    
    import asyncio
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    
    async def stream_generate():
        engine_args = AsyncEngineArgs(model="./Qwen2.5-0.5B-Instruct")
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        params = SamplingParams(temperature=0.7, max_tokens=200)
        
        # add_request 返回一个异步生成器
        results_generator = engine.generate("讲个故事", params, request_id="req-1")
        
        async for request_output in results_generator:
            # 每次迭代都能拿到目前为止生成的所有 token
            text = request_output.outputs[0].text
            print(f"\\r{text}", end="", flush=True)
        print()
    
    asyncio.run(stream_generate())
""")


# =============================================
# 3. LoRA 适配器动态加载
# =============================================
#
# 【场景】你有一个基座模型，针对不同任务微调了多个 LoRA 适配器。
# vLLM 支持在运行时动态加载/切换 LoRA，不需要重新加载基座模型！
#
# 这对多租户服务非常有用：
# 同一个基座模型 + 不同 LoRA = 不同的定制化助手

def exercise_lora():
    """
    LoRA 动态加载示例。
    
    前提：你需要有已训练好的 LoRA 权重。
    常见来源：
    - 自己用 Peft/LoRA 微调的
    - 从 HuggingFace 下载的 LoRA 适配器
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    
    print("=" * 60)
    print("LoRA 动态加载示例")
    print("=" * 60)
    
    # 加载基座模型时，开启 LoRA 支持
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        enable_lora=True,               # 开启 LoRA 支持
        max_loras=4,                     # 最多同时加载 4 个 LoRA
        max_lora_rank=64,                # LoRA 最大秩
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
    
    # 创建 LoRA 请求
    # lora_name: 给这个 LoRA 一个名字
    # lora_int_id: 唯一整数 ID
    # lora_local_path: LoRA 权重的本地路径
    lora_request = LoRARequest(
        lora_name="my-custom-lora",
        lora_int_id=1,
        lora_local_path="/path/to/your/lora/weights",  # 替换为你的路径
    )
    
    # 使用 LoRA 进行推理
    outputs = llm.generate(
        ["请写一段产品描述："],
        sampling_params,
        lora_request=lora_request,  # 指定使用哪个 LoRA
    )
    print(f"  使用 LoRA: {outputs[0].outputs[0].text}")
    
    # 不使用 LoRA（使用基座模型）
    outputs_base = llm.generate(
        ["请写一段产品描述："],
        sampling_params,
        # 不传 lora_request，就使用基座模型
    )
    print(f"  不用 LoRA: {outputs_base[0].outputs[0].text}")
    
    print("""
    
    提示：
    - 多个请求可以同时使用不同的 LoRA
    - vLLM 会自动管理 LoRA 适配器的加载和卸载
    - 在 API 服务模式下，可以通过 model 参数指定 LoRA：
      curl ... -d '{"model": "my-custom-lora", ...}'
    """)


# =============================================
# 4. 量化模型推理
# =============================================
#
# 【什么是量化？】
# 把模型权重从高精度（FP16, 每个参数2字节）压缩到低精度（如 INT4, 每个参数0.5字节）
# 好处：显存占用减少 3~4 倍，推理速度可能更快
# 代价：精度会有一点点损失（通常可以忽略）

def exercise_quantization():
    """
    量化模型推理示例。
    """
    from vllm import LLM, SamplingParams
    
    print("=" * 60)
    print("量化模型推理")
    print("=" * 60)
    
    print("""
vLLM 支持的量化格式：

┌──────────┬──────────┬──────────────────────────────────┐
│ 格式      │ 精度     │ 说明                             │
├──────────┼──────────┼──────────────────────────────────┤
│ AWQ      │ INT4     │ 最流行，精度好，速度快             │
│ GPTQ     │ INT4/8   │ 经典方案，兼容性好                │
│ FP8      │ FP8      │ 新一代，H100/4090 硬件原生支持    │
│ SqueezeLLM│ INT4    │ 非均匀量化，精度更好               │
│ BitsAndBytes│ INT4/8│ HuggingFace 生态常用              │
└──────────┴──────────┴──────────────────────────────────┘

使用示例：
""")
    
    # AWQ 量化模型
    print("1. AWQ 量化模型:")
    print("""
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        quantization="awq",
    )
    # 显存占用从 ~14GB 降到 ~4GB！
""")
    
    # GPTQ 量化模型
    print("2. GPTQ 量化模型:")
    print("""
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        quantization="gptq",
    )
""")
    
    # FP8 量化
    print("3. FP8 量化（需要 H100/4090 等支持 FP8 的显卡）:")
    print("""
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        quantization="fp8",
    )
""")
    
    # 显存对比
    print("显存占用对比（以 7B 模型为例）:")
    print("""
    FP16（原始）:  ~14 GB
    FP8:          ~7 GB
    INT4 (AWQ):   ~4 GB
    INT4 (GPTQ):  ~4 GB
    
    结论：量化是在有限显存下跑大模型的利器！
""")


# =============================================
# 5. 前缀缓存（Prefix Caching）
# =============================================
#
# 【场景】很多请求有相同的前缀（比如相同的 system prompt）
# 如果每个请求都重新计算前缀的 KV Cache，太浪费了！
# Prefix Caching 让相同前缀只计算一次。

def exercise_prefix_caching():
    """
    前缀缓存示例。
    适用于大量请求共享相同前缀的场景。
    """
    from vllm import LLM, SamplingParams
    import time
    
    print("=" * 60)
    print("前缀缓存（Prefix Caching）")
    print("=" * 60)
    
    print("""
【适用场景】
  - 所有请求都有相同的 system prompt
  - RAG 应用中，多个问题对应同一段检索到的上下文
  - Few-shot 场景，所有请求共享同样的示例

【使用方法】
  只需要在初始化时开启：
""")
    
    # 开启前缀缓存
    llm = LLM(
        model="./Qwen2.5-0.5B-Instruct",
        enable_prefix_caching=True,  # 就这一个参数！
        gpu_memory_utilization=0.7, max_model_len=2048,
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    
    # 模拟 RAG 场景：相同的长上下文 + 不同的问题
    long_context = """
    以下是一篇关于量子计算的科普文章：
    量子计算是一种基于量子力学原理的新型计算范式。与经典计算机使用比特（0或1）
    不同，量子计算机使用量子比特（qubit），它可以同时处于0和1的叠加态。这种特性
    使得量子计算机在某些特定问题上具有指数级的计算优势。量子纠缠是另一个关键概念，
    它允许两个量子比特之间存在瞬时的关联，无论它们相距多远。目前，IBM、Google、
    微软等科技巨头都在积极推进量子计算的研究。2019年，Google宣布实现了"量子霸权"，
    即量子计算机在特定任务上超越了最强大的经典超级计算机。
    
    基于以上文章回答问题：
    """
    
    questions = [
        "量子计算和经典计算的主要区别是什么？",
        "什么是量子纠缠？",
        "哪些公司在研究量子计算？",
        "Google在2019年宣布了什么成就？",
        "量子比特有什么特殊性质？",
    ]
    
    # 第一次运行（需要计算前缀的 KV Cache）
    prompts = [long_context + q for q in questions]
    
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    first_run = time.time() - start
    
    print(f"\n第一次运行（冷启动）: {first_run:.2f} 秒")
    
    # 第二次运行（前缀的 KV Cache 已被缓存）
    new_questions = [
        "量子霸权是什么意思？",
        "量子计算有什么优势？",
    ]
    new_prompts = [long_context + q for q in new_questions]
    
    start = time.time()
    outputs = llm.generate(new_prompts, sampling_params)
    second_run = time.time() - start
    
    print(f"第二次运行（前缀已缓存）: {second_run:.2f} 秒")
    print(f"加速比: {first_run / second_run:.1f}x")
    
    for output in outputs:
        print(f"\n  问题: {output.prompt[-30:]}...")
        print(f"  回答: {output.outputs[0].text[:100]}...")


# =============================================
# 6. 结构化输出（Guided Decoding）
# =============================================
#
# 让模型输出符合特定格式（如 JSON），而不是自由发挥。
# vLLM 支持通过正则表达式或 JSON Schema 来约束输出格式。

def exercise_guided_decoding():
    """
    结构化输出示例。
    让模型输出严格符合指定的 JSON Schema。
    """
    from vllm import LLM, SamplingParams
    
    print("=" * 60)
    print("结构化输出（Guided Decoding）")
    print("=" * 60)
    
    llm = LLM(model="./Qwen2.5-0.5B-Instruct",
              gpu_memory_utilization=0.7, max_model_len=2048)
    
    # 方式 1：用 JSON Schema 约束输出
    import json
    
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "city": {"type": "string"},
            "hobbies": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["name", "age", "city", "hobbies"]
    }
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=200,
    )
    
    # 通过 structured_outputs 参数控制输出格式（vllm 0.17+ 新写法）
    from vllm.sampling_params import StructuredOutputsParams

    guided_params = StructuredOutputsParams(json=json_schema)
    sampling_params.structured_outputs = guided_params
    
    prompt = "请生成一个虚构人物的信息，包含姓名、年龄、城市和爱好。"
    outputs = llm.generate([prompt], sampling_params)
    
    result = outputs[0].outputs[0].text
    print(f"\n  模型输出（保证是有效JSON）:")
    print(f"  {result}")
    
    # 验证确实是有效 JSON
    try:
        parsed = json.loads(result)
        print(f"\n  解析成功！")
        print(f"  姓名: {parsed.get('name')}")
        print(f"  年龄: {parsed.get('age')}")
        print(f"  城市: {parsed.get('city')}")
        print(f"  爱好: {parsed.get('hobbies')}")
    except json.JSONDecodeError as e:
        print(f"  JSON 解析失败: {e}")
    
    # 方式 2：用正则表达式约束
    print("\n正则约束示例:")
    print("""
    guided_params = GuidedDecodingParams(
        regex=r"(\\d{4})-(\\d{2})-(\\d{2})"  # 只能输出日期格式
    )
    """)
    
    # 方式 3：选项约束
    print("选项约束示例:")
    print("""
    guided_params = GuidedDecodingParams(
        choice=["积极", "消极", "中性"]  # 只能从这三个里选
    )
    """)


# =============================================
# 7. 多模态模型（Vision Language Model）
# =============================================
#
# vLLM 支持多模态大模型，能同时处理图片和文本。

def exercise_multimodal():
    """
    多模态模型推理示例。
    """
    print("=" * 60)
    print("多模态模型（Vision Language Model）")
    print("=" * 60)
    
    print("""
vLLM 支持的多模态模型：
  - Qwen2-VL / Qwen2.5-VL
  - LLaVA
  - InternVL
  - Phi-3-Vision
  - 等等

示例代码（以 Qwen2.5-VL 为例）：
""")
    
    print("""
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    # 多模态模型可能需要更多显存
    max_model_len=4096,
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=500)

# 方式 1：本地图片
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "text", "text": "请描述这张图片的内容。"},
        ],
    }
]

outputs = llm.chat([messages], sampling_params)
print(outputs[0].outputs[0].text)

# 方式 2：通过 API 服务
# 启动服务后，用 OpenAI SDK 的多模态接口:

from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

# 读取本地图片并转为 base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                },
                {"type": "text", "text": "这张图片里有什么？"}
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0].message.content)
""")


# =============================================
# 主程序
# =============================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║     vLLM 第四阶段：进阶使用              ║
╠══════════════════════════════════════════╣
║  1. 多卡推理（Tensor Parallel）          ║
║  2. 流式输出说明                         ║
║  3. LoRA 动态加载                        ║
║  4. 量化模型推理                         ║
║  5. 前缀缓存（Prefix Caching）           ║
║  6. 结构化输出（Guided Decoding）        ║
║  7. 多模态模型                           ║
╚══════════════════════════════════════════╝

注意：部分练习需要特定硬件/模型，
      如果条件不满足可以先阅读代码理解用法。
    """)
    
    choice = input("请选择练习编号 (1-7): ").strip()
    
    exercises = {
        "1": exercise_tensor_parallel,
        "2": exercise_streaming_offline,
        "3": exercise_lora,
        "4": exercise_quantization,
        "5": exercise_prefix_caching,
        "6": exercise_guided_decoding,
        "7": exercise_multimodal,
    }
    
    if choice in exercises:
        exercises[choice]()
    else:
        print("无效选择。以下是各功能的说明（不需要GPU）：")
        exercise_streaming_offline()
        exercise_quantization()
