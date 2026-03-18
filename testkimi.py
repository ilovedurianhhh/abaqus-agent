import os
from openai import OpenAI

# 1. 配置你的 Kimi API
client = OpenAI(
    api_key="sk-Yu9eFrSMdiYjo7WuSJi7AyEnAdapafrp3rc6XmSvPRfpXPmH", 
    base_url="https://api.moonshot.cn/v1",
)

def test_api_full():
    # --- 测试 1: Chat 对话能力 ---
    try:
        print("1. 正在测试 Chat (对话) 接口...")
        chat_response = client.chat.completions.create(
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "你好，请回复'连接成功'"}]
        )
        print(f"✅ Chat 响应: {chat_response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Chat 测试失败: {e}")

    print("-" * 30)


if __name__ == "__main__":
    test_api_full()