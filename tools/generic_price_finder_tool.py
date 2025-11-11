# -*- coding: utf-8 -*-
"""
generic_price_finder_tool.py — v2.5（自然语言识别输入 + 中国+海外双检索）
"""

import os
import ssl
import httpx
import json
import re
import requests
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


class GenericPriceFinderArgs(BaseModel):
    material_name: str = Field(..., description="Material name or natural-language query (e.g., '计算 Loctite 603 价格')")


class GenericPriceFinderTool:
    def __init__(self, llm=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.abspath(os.path.join(base_dir, "..", ".env"))
        load_dotenv(dotenv_path=env_path)

        self.name = "generic_price_finder"
        self.description = "Search industrial adhesive prices and estimate price (CNY/gram)."
        self.llm = llm
        self.proxy = os.getenv("PROXY_URL")
        self.tavily_key = os.getenv("TAVILY_API_KEY", "")
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

        # 环境代理配置
        os.environ["HTTP_PROXY"] = self.proxy or ""
        os.environ["HTTPS_PROXY"] = self.proxy or ""
        self.requests_proxies = {"http": self.proxy, "https": self.proxy}
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.options |= ssl.OP_NO_TLSv1_3
        self.client = httpx.Client(proxy=self.proxy, verify=False, trust_env=True, timeout=30.0)

    # ======================================================
    # Step 1. 自动识别材料名
    # ======================================================
    def _extract_material_name(self, text: str) -> str:
        """从自然语言中提取材料名称，如 'Loctite 603'"""
        # 简单正则匹配：提取连续的英文字母+数字组合
        match = re.search(r'([A-Za-z]+[-]?\s*\d+[A-Za-z0-9]*)', text)
        if match:
            return match.group(1).strip()

        # 若正则失败，使用 LLM 尝试提取
        try:
            llm = AzureChatOpenAI(
                deployment_name="gpt-5",
                api_key=self.azure_key,
                azure_endpoint=self.azure_endpoint,
                api_version="2025-01-01-preview",
                http_client=self.client,
            )
            prompt = ChatPromptTemplate.from_template("""
            Extract the product or material name from this sentence. 
            Only return the exact name (e.g., "Loctite 603").
            Sentence: {text}
            """)
            return (prompt | llm).invoke({"text": text}).content.strip()
        except Exception:
            return text

    # ======================================================
    # Step 2. 汇率
    # ======================================================
    def _get_usd_to_cny_rate(self) -> float:
        try:
            res = requests.get(
                "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
                proxies=self.requests_proxies,
                verify=False,
                timeout=10,
            )
            return float(res.json()["usd"]["cny"])
        except Exception:
            print("[WARN] 汇率获取失败，使用默认 7.12")
            return 7.12

    # ======================================================
    # Step 3. Tavily 搜索
    # ======================================================
    def _search_china_market(self, name: str) -> str:
        print(f"🔍 Tavily（中国）搜索: {name} + 工业用胶水")
        try:
            search = TavilySearch(api_key=self.tavily_key, max_results=5, client=self.client)
            results = search.invoke(f"{name} 工业用胶水 中国 市场 价格")
            return self._format_results(results)
        except Exception as e:
            print(f"❌ Tavily 中国查询失败: {e}")
            return ""

    def _search_overseas_market(self, name: str) -> str:
        print(f"🌍 Tavily（海外）搜索: {name} + industrial glue")
        try:
            search = TavilySearch(api_key=self.tavily_key, max_results=5, client=self.client)
            results = search.invoke(f"{name} industrial glue price")
            return self._format_results(results)
        except Exception as e:
            print(f"❌ Tavily 海外查询失败: {e}")
            return ""

    @staticmethod
    def _format_results(results) -> str:
        if isinstance(results, list):
            return "\n".join(f"🔗 {r.get('title')}\n🌍 {r.get('url')}\n{r.get('content')}" for r in results)
        if isinstance(results, dict) and "results" in results:
            return "\n".join(f"🔗 {r.get('title')}\n🌍 {r.get('url')}\n{r.get('content')}" for r in results["results"])
        return str(results)

    # ======================================================
    # Step 4. LLM 提取价格
    # ======================================================
    def _extract_price(self, china: str, overseas: str, name: str, rate: float) -> Dict[str, Any]:
        llm = AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=self.azure_key,
            azure_endpoint=self.azure_endpoint,
            api_version="2025-01-01-preview",
            http_client=self.client,
        )
        prompt = ChatPromptTemplate.from_template("""
        You are a procurement analyst. The product is industrial adhesive.
        Goal:
        1. Extract price information from BOTH China and overseas search results if possible.
        2. Convert price to CNY/gram when needed.
           - Example: 50 ml bottle ≈ 50 g.
           - If price is in USD, convert using 1 USD = {usd_to_cny:.3f} CNY.
        3. Selection rule:
           - If China has numeric price → final_region = "china"
           - Else if only overseas has → final_region = "overseas"
        Return ONLY JSON:
        {{
          "china": {{"price_cny_per_g": <num or null>, "source": "<url or null>"}},
          "overseas": {{"price_cny_per_g": <num or null>, "source": "<url or null>"}},
          "final_region": "china" | "overseas" | null,
          "final_price_cny_per_g": <num or null>,
          "reasoning": "<why>"
        }}
        China Results:
        {china_content}

        Overseas Results:
        {overseas_content}
        """)
        result = (prompt | llm).invoke({
            "material_name": name,
            "china_content": china or "",
            "overseas_content": overseas or "",
            "usd_to_cny": rate
        }).content.strip()

        match = re.search(r'({.*})', result, re.DOTALL)
        clean = match.group(1) if match else result
        try:
            return json.loads(clean)
        except Exception:
            return {
                "china": {"price_cny_per_g": None, "source": None},
                "overseas": {"price_cny_per_g": None, "source": None},
                "final_region": None,
                "final_price_cny_per_g": None,
                "reasoning": "LLM parse error"
            }

    # ======================================================
    # Step 5. 主执行逻辑
    # ======================================================
    def run(self, user_input: str) -> str:
        print(f"\n[INFO] 收到输入: {user_input}")
        material_name = self._extract_material_name(user_input)
        print(f"[INFO] 提取材料名: {material_name}")

        rate = self._get_usd_to_cny_rate()
        china_raw = self._search_china_market(material_name)
        overseas_raw = self._search_overseas_market(material_name)
        extracted = self._extract_price(china_raw, overseas_raw, material_name, rate)

        return json.dumps({
            "input": user_input,
            "material_name": material_name,
            "final_price_cny_per_g": extracted.get("final_price_cny_per_g"),
            "final_region": extracted.get("final_region"),
            "reasoning": extracted.get("reasoning"),
            "china": {
                "price_cny_per_g": extracted.get("china", {}).get("price_cny_per_g"),
                "source": extracted.get("china", {}).get("source"),
                "raw": china_raw[:500]
            },
            "overseas": {
                "price_cny_per_g": extracted.get("overseas", {}).get("price_cny_per_g"),
                "source": extracted.get("overseas", {}).get("source"),
                "raw": overseas_raw[:500]
            }
        }, ensure_ascii=False, indent=2)

    # ======================================================
    # Step 6. LangChain Tool 封装
    # ======================================================
    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=GenericPriceFinderArgs,
        )


if __name__ == "__main__":
    tool = GenericPriceFinderTool()
    print(tool.run("计算 Loctite 603 价格"))
