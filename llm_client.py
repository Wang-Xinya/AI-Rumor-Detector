"""
llm_client.py
LLM 客户端配置文件 —— 接入API
"""

import requests
import time

BASE_URL = "https://models.sjtu.edu.cn/api/v1"   # API endpoint
API_KEY  = "sk-lENrTfpXuHSIjQ8uCv6imQ"           # API key
MODEL    = "deepseek-chat"           # 模型名（目前使用的是 DeepSeek V3.2 常规模式，可以尝试思考模式或别的模型）

CONFIG = {
    "base_url": BASE_URL,
    "api_key":  API_KEY,
    "model":    MODEL,
    "temperature": 1.0,    # 可调参
    "top_p":       1.0,
    "max_tokens":  8192,
}

def call_llm(messages: list[dict], retries: int = 2) -> str:
    """调用 LLM，返回回复文本"""
    url = f"{BASE_URL}/chat/completions"          # 完整 URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "model": MODEL,
        "messages": messages,                     # 直接使用传入的 messages
        "stream": False,
        "temperature": CONFIG["temperature"],
        "top_p": CONFIG["top_p"],
        "max_tokens": CONFIG["max_tokens"],
    }

    last_exception = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=(30, 600))
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            # 非 200：打印错误，决定是否重试
            print(f"HTTP {resp.status_code} (尝试 {attempt+1}/{retries+1}): {resp.text[:500]}")
            # 4xx 客户端错误（除 429 外）通常重试无用，直接退出循环
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break
        except Exception as e:
            last_exception = e
            print(f"请求异常 (尝试 {attempt+1}/{retries+1}): {e}")

        # 重试前等待（指数退避）
        if attempt < retries:
            time.sleep(1.0 * (attempt + 1))

    raise last_exception or RuntimeError("LLM 调用失败，未见具体异常")


if __name__ == "__main__":
    print("Testing LLM connection...")
    try:
        result = call_llm([{"role": "user", "content": "Say 'hello' in one word."}])
        print(f"✓ Connected. Response: {result[:100]}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("请检查 llm_client.py 中的 CONFIG")
