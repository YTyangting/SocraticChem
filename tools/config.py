import os

class Config:
    # 1. 基础 API 配置
    # 建议从环境变量读取，这里提供默认值作为示例
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_key")
    OPENAI_BASE_URL = "https://api.vveai.com/v1"

    # 2. Agent 模型分配策略
    
    # [Teacher]: 需要极强的逻辑推理、教学策略调整和长上下文理解能力
    # 推荐: gpt-4o, gpt-4-turbo
    MODEL_TEACHER = "gpt-4.1" 

    # [Student]: 主要进行角色扮演 (Roleplay)，不需要太强的逻辑，速度要快
    # 推荐: gpt-4o-mini, gpt-3.5-turbo
    MODEL_STUDENT = "gpt-4.1-mini"

    # [Translator]: 将自然语言转为严格的 JSON 动作指令，需要极强的指令遵循能力 (Instruction Following)
    # 推荐: gpt-4o (为了保证 JSON 格式不出错，建议用强模型)
    MODEL_TRANSLATOR = "gpt-4.1"

    # [Reviewer/Diagnosis]: 需要进行深度归因分析，判断是“加多了”还是“反应了”，逻辑要求高
    # 推荐: gpt-4o
    MODEL_REVIEWER = "gpt-4.1"
    # [Global]: 超时与重试设置
    TIMEOUT = 30.0
    MAX_RETRIES = 3
    # 场景切换开关
    CURRENT_PROVIDER = "local_vllm"  # 可选: "openai", "deepseek", "local_vllm", "ollama"

    PROVIDERS = {
        "local_vllm": {
            "api_key": "EMPTY", # vLLM 通常不需要 key
            "base_url": "http://localhost:8000/v1",
            "model": "qwen2.5-7b-socratic-finetuned" # 你微调的模型名
        },
        "ollama": {
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b"
        }
    }