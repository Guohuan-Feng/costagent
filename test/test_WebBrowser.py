# -*- coding: utf-8 -*-
"""
Tavily + Azure GPT-5 版
✅ 兼容 Bosch 网络，自动 fallback（仅使用 Tavily 内容）
"""

import os
import ssl
import httpx
import json
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# =========================================================
# ✅ Step 1. 载入环境变量
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)
print(f"✅ 已加载 .env 文件: {env_path}")

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
PROXY = os.getenv("PROXY_URL")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not all([TAVILY_KEY, AZURE_KEY, AZURE_ENDPOINT]):
    raise RuntimeError("❌ .env 缺少必要变量，请检查。")

print(f"🌐 当前代理: {PROXY or '系统默认'}")

# =========================================================
# ✅ Step 2. 启用 TLS1.2（兼容 Bosch）
# =========================================================
ctx = ssl.create_default_context()
ctx.options |= ssl.OP_NO_TLSv1_3
ctx.check_hostname = True

transport = httpx.HTTPTransport()
transport._ssl_context = ctx

# =========================================================
# ✅ Step 3. 创建共享 httpx 客户端（供 Azure / Tavily 使用）
# =========================================================
shared_client = httpx.Client(
    proxy=PROXY,
    trust_env=True,
    verify=True,
    timeout=30.0,
    transport=transport
)

# =========================================================
# ✅ Step 4. 初始化 Azure GPT-5 模型
# =========================================================
llm = AzureChatOpenAI(
    deployment_name="gpt-5",
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview",
    http_client=shared_client
)

# =========================================================
# ✅ Step 5. Tavily 搜索 + 自动 fallback
# =========================================================
search = TavilySearch(api_key=TAVILY_KEY, max_results=5)
query = (
    "Latest China domestic price per kg (CNY) for silicon "
    "site:metal.com OR site:sunsirs.com OR site:tradingeconomics.com"
)
print(f"🔍 正在使用 Tavily 搜索: {query}")

raw_result = search.invoke(query)
print("✅ Tavily 搜索完成。")

# --- 尝试解析 JSON 返回 ---
try:
    results = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
except Exception:
    results = []
    print("⚠️ Tavily 返回数据无法解析为 JSON。")

# --- 合并 Tavily 内容 ---
combined_text = ""
if isinstance(results, list):
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("content", "")
        combined_text += f"🔗 {title}\n🌍 {url}\n{snippet}\n\n"
else:
    combined_text = str(results)

if not combined_text.strip():
    raise RuntimeError("❌ Tavily 未返回任何文本内容，请检查 API 密钥或网络。")

# =========================================================
# ✅ Step 6. 让模型提取价格
# =========================================================
prompt = ChatPromptTemplate.from_template("""
You are an analytical assistant. From the following web summaries,
extract the *latest China domestic silicon price per kilogram (CNY/kg)*.
If multiple numbers appear, choose the most recent and clearly stated one.
Return only the numeric value or range and a short summary with date/source.

Web Summaries:
{content}
""")

chain = RunnableSequence(prompt | llm)
print("🧠 正在分析 Tavily 搜索结果并提取价格...\n")
result = chain.invoke({"content": combined_text})

print("\n🔎 提取结果:\n", result.content)
