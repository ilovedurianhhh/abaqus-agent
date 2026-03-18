#!/usr/bin/env python3
"""Simple script to test Kimi API connectivity and authentication."""

import os
import sys
from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.environ.get("KIMI_API_KEY")
if not api_key:
    print("[ERROR] .env 文件中未找到 KIMI_API_KEY")
    sys.exit(1)

print(f"[OK] 找到 API 密钥: {api_key[:10]}...{api_key[-10:]}")

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] 未安装 openai 库")
    print("   请运行: pip install openai")
    sys.exit(1)

print("\n正在测试 Kimi API 连接...\n")

try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1"
    )

    print("[OK] 创建客户端成功")
    print("[INFO] 发送测试请求...\n")

    response = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[
            {"role": "user", "content": "你好，请简短回答：1+1等于多少?"}
        ],
        max_tokens=100,
    )

    print("[OK] API 响应成功!")
    print(f"\n模型: {response.model}")
    print(f"回答: {response.choices[0].message.content}")
    print(f"\n用量统计:")
    print(f"  - 输入tokens: {response.usage.prompt_tokens}")
    print(f"  - 输出tokens: {response.usage.completion_tokens}")
    print(f"  - 总tokens: {response.usage.total_tokens}")
    print("\n[SUCCESS] Kimi API 正常工作!")

except Exception as e:
    print(f"[ERROR] API 测试失败:")
    print(f"   错误类型: {type(e).__name__}")
    print(f"   错误信息: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
