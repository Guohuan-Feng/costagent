# =====================================
# ✅ test_agent.py — Bosch 代理 + Azure 模型版本检测 + LangGraph 测试（智能分流版）
# =====================================
import sys, os, warnings, httpx, json, time, traceback
from dotenv import load_dotenv
from perf_aop import aop_inject_timing

start_all = time.time()
warnings.filterwarnings("ignore")

# 1️⃣ 路径设置
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 2️⃣ 自动加载 .env
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)
print(f"✅ 已加载 .env 文件: {env_path}")

# 3️⃣ 环境变量与代理设置
proxy = os.getenv("PROXY_URL", "").strip()
tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()

if tavily_key:
    print(f"✅ 检测到 Tavily API Key: {tavily_key[:15]}... (已隐藏后半部分)")
else:
    print("⚠️ 未检测到 TAVILY_API_KEY")

# 🔧 Bosch 代理强制启用
proxy = proxy or "http://fun4wx:qawaearata0A%21@rb-proxy-unix-szh.bosch.com:8080"
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["ALL_PROXY"] = proxy
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
print("✅ 强制全局代理启用:", proxy)

# 4️⃣ 测试 Azure Endpoint 连通性
can_access_azure = False
if azure_endpoint and azure_api_key:
    try:
        print(f"🔍 测试 Azure Endpoint: {azure_endpoint}")
        base_url = azure_endpoint.split("/openai/")[0]
        with httpx.Client(proxy=proxy, verify=False, trust_env=True, timeout=10.0) as client:
            resp = client.get(base_url)
            print("🔗 Azure HTTP 测试状态码:", resp.status_code)
            if resp.status_code < 500:
                can_access_azure = True
    except Exception as e:
        print("⚠️ Azure 测试失败:", e)
else:
    print("⚠️ AZURE_OPENAI_ENDPOINT 或 AZURE_OPENAI_API_KEY 未配置")
print("🌐 Azure 可访问？", can_access_azure)

# ✅ 4.5️⃣ 检查部署模型
if can_access_azure:
    try:
        print("\n🔎 查询部署 'gpt-5' 实际模型...\n")
        deployment_name = "gpt-5"
        api_url = f"{azure_endpoint}openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview"
        payload = {"messages": [{"role": "user", "content": "Say hello"}], "temperature": 1.0}
        headers = {"api-key": azure_api_key, "Content-Type": "application/json"}
        with httpx.Client(proxy=proxy, verify=False, trust_env=True, timeout=20.0) as client:
            r = client.post(api_url, headers=headers, json=payload)
            if r.status_code == 200:
                print(f"📦 模型版本: {r.json().get('model', '(unknown)')}")
            else:
                print(f"⚠️ 查询失败，HTTP {r.status_code}")
    except Exception as e:
        print("⚠️ 查询模型名称失败:", e)

# ✅ 5️⃣ 测试 GPT-5 调用
if can_access_azure:
    try:
        print("\n🧠 测试 Azure GPT-5 模型调用...\n")
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version="2025-01-01-preview",
            temperature=1.0,
        )
        start_time = time.time()
        resp = llm.invoke("Reply with only: OK")
        print(f"✅ 模型回复: {resp.content.strip()}")
        print(f"⏱️ 调用耗时: {time.time() - start_time:.2f} 秒")
    except Exception as e:
        print("❌ GPT-5 模型调用失败:", e)
else:
    print("⚠️ 跳过 GPT-5 调用测试（Azure 不可访问）")

# ===========================================================
# ✅ 智能识别逻辑：合金 → LangGraph；非合金 → GenericPriceFinderTool
# ===========================================================
from agent import chain, save_json_report, parse_user_inputs_from_query
from tools.generic_price_finder_tool import GenericPriceFinderTool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

def is_alloy_query(user_query: str) -> bool:
    """用 GPT 判断是否为合金类材料"""
    try:
        llm = AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version="2025-01-01-preview",
            temperature=1.0,
        )
        prompt = f"""
        判断以下输入是否涉及金属合金（例如 AlSi9Mn、CuAl10Ni2、AA6061、FeCrNi 等）。
        如果是合金，请只输出 “YES”；否则输出 “NO”。
        输入：{user_query}
        """
        resp = llm.invoke(prompt)
        answer = resp.content.strip().upper()
        print(f"🧩 GPT 判断结果：{answer}")
        return "YES" in answer
    except Exception as e:
        print("⚠️ GPT 判断失败，默认认为是合金。错误：", e)
        return True

def run_smart_query(user_query: str):
    """根据 query 自动判断执行路径"""
    print("\n==================== 智能识别执行 ====================\n")
    print(f"🧠 用户输入: {user_query}\n")

    if is_alloy_query(user_query):
        print("✅ 判断为合金 → 执行 LangGraph 主链")
        start = time.time()
        result = chain.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": "smart-auto"}}
        )
        print(f"⏱️ LangGraph 执行耗时: {time.time() - start:.2f} 秒\n")

        # 输出结果与保存
        for m in result.get("messages", []):
            print(f"[{m.__class__.__name__}] 内容预览:\n{m.content[:400]}...\n")
        save_json_report({
            "messages": result.get("messages", []),
            "user_inputs": parse_user_inputs_from_query(user_query)
        })
        print("✅ 已保存到本地/云端。")
        return result
    else:
        print("✅ 判断为非合金 → 执行 GenericPriceFinderTool")
        tool = GenericPriceFinderTool()
        start = time.time()
        output = tool.run(user_query)
        print(f"⏱️ GenericPriceFinderTool 执行耗时: {time.time() - start:.2f} 秒\n")
        print("输出结果预览（前 800 字符）:\n", output[:800])
        return output

# ===========================================================
# ✅ 单一 query 测试入口（自动判断执行）
# ===========================================================
# ⚠️ 只改这一行即可：如果换成 Loctite 603 会自动识别为非合金
query = "calculate this material AlSi9Mn price; Location=Ningbo, Zhejiang; supplier_code=97036203; part_number=044220003G; sub_process_step=AlSi9Mn; process_type=raw_material"
# query = "Find the price of Loctite 603"

print("\n==================== 智能测试开始 ====================\n")
try:
    run_smart_query(query)
except Exception as e:
    print(f"❌ 执行出错: {e}")
    traceback.print_exc()

print(f"\n🕒 程序总耗时: {time.time() - start_all:.2f} 秒")
