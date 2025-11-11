# -*- coding: utf-8 -*-
"""
✅ Tavily + Azure GPT-5 (Bosch 内网稳定版·合金估价版)
- 自动查 aluminum / silicon / manganese 价格
- 根据 alloy_name (AlSi9Mn) 计算加权平均合金价
"""

import os, json, httpx
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# =========================================================
# Step 1. 环境初始化
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
# 当前文件所在路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# ✅ 向上一层目录查找 .env
env_path = os.path.join(base_dir, "..", ".env")
env_path = os.path.abspath(env_path)  # 规范化为绝对路径

# ✅ 加载 .env
load_dotenv(dotenv_path=env_path)
print(f"✅ 已加载 .env 文件: {env_path}")

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
PROXY = os.getenv("PROXY_URL")

if not all([TAVILY_KEY, AZURE_KEY, AZURE_ENDPOINT, PROXY]):
    raise RuntimeError("❌ .env 缺少必要变量，请检查。")

os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# =========================================================
# Step 2. Tavily & GPT 初始化
# =========================================================
search = TavilySearch(api_key=TAVILY_KEY, max_results=5)
http_client = httpx.Client(proxy=PROXY, verify=False, timeout=30.0)
llm = AzureChatOpenAI(
    deployment_name="gpt-5",
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview",
    http_client=http_client,
)

# =========================================================
# Step 3. LLM 提取模板
# =========================================================
prompt = ChatPromptTemplate.from_template("""
You are a financial and materials analyst.
From the following Tavily search results, extract the *latest China domestic market price per kilogram (CNY/kg)* for {material}.
If multiple prices appear, choose the most recent one.
Convert CNY/ton to CNY/kg if needed.
Output strictly in JSON:
{{
  "currency": "CNY",
  "name": "{material}",
  "price_per_unit": <numeric>,
  "source": "<url>",
  "notes": "<short explanation>",
  "unit": "kg"
}}

Web Summaries:
{content}
""")

# =========================================================
# Step 4. 查询函数
# =========================================================
def get_material_price(material: str):
    print(f"\n🔍 正在搜索 {material} 价格...")
    query = f"Latest China domestic price per kg (CNY) for {material}"
    raw_result = search.invoke(query)
    try:
        results = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception:
        results = []

    combined_text = ""
    if isinstance(results, list):
        for r in results:
            combined_text += f"{r.get('title','')}\n{r.get('url','')}\n{r.get('content','')}\n\n"
    else:
        combined_text = str(results)

    if not combined_text.strip():
        print(f"❌ 无 {material} 数据")
        return None

    chain = RunnableSequence(prompt | llm)
    result = chain.invoke({"material": material, "content": combined_text})
    try:
        data = json.loads(result.content)
        data["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return data
    except Exception:
        return {"name": material, "notes": result.content}

# =========================================================
# Step 5. 查询各元素
# =========================================================
materials = ["aluminum", "silicon", "manganese"]
data_map = {m: get_material_price(m) for m in materials}

# =========================================================
# Step 6. 根据 alloy_name 配比计算总价
# =========================================================
# 典型 AlSi9Mn 配比
composition = {"aluminum": 0.905, "silicon": 0.09, "manganese": 0.005}

def calc_alloy_price(composition, prices):
    total = 0.0
    valid = 0
    for el, ratio in composition.items():
        price = prices.get(el, {}).get("price_per_unit")
        if price:
            total += ratio * price
            valid += ratio
    return round(total / valid, 2) if valid > 0 else None

alloy_price = calc_alloy_price(composition, data_map)
alloy_name = "AlSi9Mn"

# =========================================================
# Step 7. 输出 JSON 结果
# =========================================================
final_output = {
    "materials": data_map,
    "alloy": {
        "alloy_name": alloy_name,
        "composition": composition,
        "estimated_price_CNY_per_kg": alloy_price,
        "calculated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }
}

print("\n✅ 最终结构化输出:\n")
print(json.dumps(final_output, indent=2, ensure_ascii=False))
