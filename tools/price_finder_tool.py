# -*- coding: utf-8 -*-
"""
price_finder_tool.py — v6.0 (Async-enabled)
✅ 并行 Tavily + GPT-5 提取金属价格，加速合金成本计算
✅ 自动汇率、代理、单位归一，兼容 Bosch 环境
"""

import os
import ssl
import httpx
import json
import re
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from tools.process_finder_tool import ProcessFinderTool


class PriceFinderArgs(BaseModel):
    user_request: str = Field(..., description="User question about material/alloy price estimation.")
    PROCESS_CONTEXT: dict | None = None  # ✅


class PriceFinderTool:
    """Integrates process_finder_tool → Tavily → GPT-5 to estimate element-wise and total alloy price (CNY/kg)."""

    def __init__(self, llm=None):
        load_dotenv()
        self.name = "price_finder"
        self.description = (
            "Generates PROCESS_CONTEXT via process_finder_tool, retrieves live USD→CNY exchange rate "
            "from fawazahmed0 currency-api, then searches web (Tavily + GPT-5) for each element price "
            "and computes weighted total price (CNY/kg)."
        )
        self.llm = llm
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.proxy = os.getenv("PROXY_URL")
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        # TLS 1.2 兼容 (Bosch)
        self.ctx = ssl.create_default_context()
        self.ctx.options |= ssl.OP_NO_TLSv1_3
        self.transport = httpx.HTTPTransport()
        self.transport._ssl_context = self.ctx
        self.client = httpx.Client(proxy=self.proxy, trust_env=True, verify=True, timeout=30.0, transport=self.transport)

    # ---------------------------- 汇率获取 ----------------------------
    def _get_usd_to_cny_rate(self) -> float:
        """从 fawazahmed0 currency-api 获取实时 USD→CNY 汇率"""
        try:
            url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
            res = requests.get(url, timeout=10)
            data = res.json()
            rate = float(data["usd"]["cny"])
            print(f"[DEBUG] ✅ 实时汇率: 1 USD = {rate:.4f} CNY")
            return rate
        except Exception as e:
            print(f"[WARN] ⚠️ 汇率获取失败 ({e})，使用默认值 7.12")
            return 7.12  # fallback 汇率

    # ---------------------------- Tavily 搜索 ----------------------------
    def _search_each_element(self, element: str) -> str:
        """使用 Tavily 搜索单个元素价格（含耗时统计）。"""
        start = time.time()
        search = TavilySearch(api_key=self.tavily_key, max_results=5)
        query = f"Latest China domestic price for {element} metal"
        print(f"🔍 Tavily 查询元素价格: {query}")

        try:
            raw_result = search.invoke(query)
            print(f"[DEBUG] Tavily 搜索结果: {raw_result}")
        except Exception as e:
            print(f"❌ Tavily 搜索 {element} 失败 ({e})，耗时: {time.time() - start:.2f} 秒")
            return ""

        print(f"⏱️ Tavily 搜索 {element} 完成，耗时: {time.time() - start:.2f} 秒")
        if not raw_result:
            return ""
        if isinstance(raw_result, list):
            return "\n".join(f"🔗 {r.get('title','')}\n🌍 {r.get('url','')}\n{r.get('content','')}" for r in raw_result)
        return str(raw_result)

    # ---------------------------- GPT-5 抽取价格 ----------------------------
    def _extract_price(self, content: str, material: str, usd_to_cny: float) -> Dict[str, Any]:
        """让 GPT-5 自行识别单位并使用传入汇率换算为 CNY/kg。"""
        llm = AzureChatOpenAI(
            deployment_name="gpt-4o-2024-08-06",
            api_key=self.azure_key,
            azure_endpoint=self.azure_endpoint,
            api_version="2025-01-01-preview",
            http_client=self.client,
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a financial analyst specializing in metal pricing.
The current exchange rate is: 1 USD = {usd_to_cny:.3f} CNY.

From the following web summaries, find the **latest quoted price** for {material},
and convert it to **CNY/kg** using that exchange rate if needed.

Conversion rules you should apply yourself:
- If the price is quoted in **CNY/ton** or **CNY/MT**, divide by 1000.
- If in **USD/MT** or **USD/ton**, multiply by {usd_to_cny:.3f}/1000.
- If in **CNY/mtu**, divide by 10.
- If already **CNY/kg**, use directly.

Return strictly in JSON:
{{
  "price_CNY_per_kg": <numeric_value_or_null>,
  "unit_detected": "CNY/kg",
  "conversion_explanation": "<how you converted it>",
  "summary": "<short summary>",
  "source": "<url_if_present>"
}}

Web summaries:
{content}
"""
        )

        chain = RunnableSequence(prompt | llm)
        result = chain.invoke({"material": material, "content": content, "usd_to_cny": usd_to_cny})
        result_content = result.content

        # 清理 markdown 包装
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', result_content, re.DOTALL | re.IGNORECASE)
        cleaned_content = match.group(1).strip() if match else result_content.strip()

        try:
            parsed = json.loads(cleaned_content)
            print(f"[DEBUG] LLM 输出内容: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
            return parsed
        except Exception as e:
            print(f"[ERROR] ❌ LLM JSON 解析失败: {e}. Raw content: {result_content[:300]}")
            return {
                "price_CNY_per_kg": None,
                "unit_detected": "unknown",
                "conversion_explanation": f"LLM parsing failed: {e}",
                "summary": f"Failed to parse: {result_content[:300]}",
                "source": None,
            }

    # ---------------------------- 新增：并行处理单个元素 ----------------------------
    def _process_element(self, elem: str, frac: float, usd_to_cny: float):
        """单线程执行 Tavily + GPT-5 完整流程"""
        try:
            start = time.time()
            content = self._search_each_element(elem)
            parsed = self._extract_price(content, elem, usd_to_cny)
            price = parsed.get("price_CNY_per_kg")
            print(f"✅ {elem} 完成，总耗时 {time.time() - start:.2f}s")
            return elem, {
                "fraction": frac,
                "query_used": f"Latest China domestic price for {elem} metal",
                **parsed,
            }
        except Exception as e:
            return elem, {"fraction": frac, "error": str(e)}

    # ---------------------------- 主流程 ----------------------------
    # ✅ 主流程（已修复）
    def run(self, user_request: str, PROCESS_CONTEXT: dict | None = None) -> str:
        print(f"[DEBUG] price_finder.run 启动: {user_request}")

        usd_to_cny = self._get_usd_to_cny_rate()
        composition, material = {}, None

        # ✅ 优先使用参数传入的 PROCESS_CONTEXT
        if PROCESS_CONTEXT:
            print("[DEBUG] ✅ 使用传入的 PROCESS_CONTEXT 参数")
            composition = PROCESS_CONTEXT.get("composition", {})
            material = PROCESS_CONTEXT.get("material")
        else:
            # ⬇️ 回退机制 (兼容旧链路)
            print("[DEBUG] ⚠️ PROCESS_CONTEXT 未作为参数传入，调用 process_finder_tool() 重新推理")
            process_tool = ProcessFinderTool()
            process_json = process_tool.run(user_request)
            try:
                data = json.loads(process_json)
                composition = data.get("composition", {})
                material = re.search(r"\b[A-Z][a-z]?[A-Za-z0-9]*\b", user_request).group(0)
            except:
                return json.dumps({"error": "process_finder_tool failed"}, ensure_ascii=False)

        if not composition:
            return json.dumps({"error": "No composition found"}, ensure_ascii=False)

        print(f"[DEBUG] ✅ 使用成分: {composition}")

        results, total = {}, 0.0
        max_workers = min(5, len(composition))
        print(f"[INFO] ⚙️ 并行 Tavily 查询: {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            future_to_elem = {
                exe.submit(self._extract_price, self._search_each_element(elem), elem, usd_to_cny): elem
                for elem in composition
            }
            for future in as_completed(future_to_elem):
                elem = future_to_elem[future]
                price_info = future.result()
                results[elem] = price_info
                if price_info.get("price_CNY_per_kg"):
                    total += price_info["price_CNY_per_kg"] * composition[elem]

        return json.dumps(
            {
                "material": material,
                "composition": composition,
                "usd_to_cny": usd_to_cny,
                "materials": results,
                "total_cost": round(total, 3) if total else None,
                "unit": "CNY/kg",
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False, indent=2
        )

                    

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=PriceFinderArgs,
        )


if __name__ == "__main__":
    tool = PriceFinderTool()
    print(tool.run("Calculate the price for AlSi9Mn alloy."))
