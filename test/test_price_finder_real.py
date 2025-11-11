import asyncio
import logging
import json
import re
import ssl
from datetime import datetime, timezone

import aiohttp
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS

# --- 配置 ---
# 设置日志记录
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# 定义要查询的材料
MATERIALS_TO_FIND = ["Aluminum", "Manganese", "Silicon"]

# 为价格合理性校验设置一个最低阈值 (单位: CNY/kg)
# 这可以防止从网站上抓取到错误的、过低的价格（如示例价、运费等）
# 根据当前市场行情估算，例如：铝 > 15, 硅 > 10, 锰 > 8
MINIMUM_REASONABLE_PRICES = {
    "Aluminum": 10.0,
    "Manganese": 8.0,
    "Silicon": 10.0,
}

# --- 主要逻辑 ---

async def find_prices():
    """
    为所有定义的材料异步查找价格。
    """
    # 创建一个 aiohttp 客户端会话
    # 下面的 SSL context 设置是为了重现您日志中的 DeprecationWarning
    # 它禁用了 TLS 1.3，以确保与旧服务器的兼容性
    ctx = ssl.create_default_context()
    try:
        # DeprecationWarning: ssl.OP_NO_SSL*/ssl.OP_NO_TLS* options are deprecated
        ctx.options |= ssl.OP_NO_TLSv1_3  # 确保保留 TLS 1.2 兼容性
    except AttributeError:
        # 在某些 Python 版本中可能不存在此选项
        logging.warning("ssl.OP_NO_TLSv1_3 not available in this Python version.")

    connector = aiohttp.TCPConnector(ssl=ctx)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 为每种材料创建一个异步任务
        tasks = [get_material_price(material, session) for material in MATERIALS_TO_FIND]
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 将结果构建成最终的字典结构
    final_output = {"materials": {}}
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"An exception occurred: {result}")
        else:
            material_name = result.get("name", "unknown").lower()
            final_output["materials"][material_name] = result
            
    return final_output

def parse_price_from_text(text, material_name):
    """
    从文本中解析价格，优先找kg，其次找ton。
    """
    # 移除千位分隔符
    text = text.replace(',', '')

    # 1. 尝试查找每千克 (kg) 的价格
    # 匹配 "数字 /kg" 或 "CNY 数字/kg" 等模式
    kg_pattern = re.compile(r'(\d{1,8}(?:\.\d{1,4})?)\s*(?:CNY|¥|元)?\s*/\s*(?:kg|千克)', re.IGNORECASE)
    kg_matches = kg_pattern.findall(text)
    if kg_matches:
        price = float(kg_matches[0])
        # 合理性检查
        if price < MINIMUM_REASONABLE_PRICES.get(material_name, 0.01):
            return None, f"Kg price {price} found but rejected as unreasonably low for {material_name}."
        return price, f"Found price per kg: {price}"

    # 2. 如果没找到kg，尝试查找每吨 (ton) 的价格
    # 匹配 "数字 /ton" 或 "CNY 数字/吨" 等模式
    ton_pattern = re.compile(r'(\d{1,8}(?:\.\d{1,4})?)\s*(?:CNY|¥|元)?\s*/\s*(?:ton|t|吨)', re.IGNORECASE)
    ton_matches = ton_pattern.findall(text)
    if ton_matches:
        ton_price = float(ton_matches[0])
        kg_price = ton_price / 1000.0
        # 合理性检查
        if kg_price < MINIMUM_REASONABLE_PRICES.get(material_name, 0.01):
            return None, f"Ton price {ton_price} found but rejected as unreasonably low for {material_name}. (Converted to kg: {kg_price:.3f})"
        return kg_price, f"Found price per ton: {ton_price}, converted to kg: {kg_price:.3f}"

    return None, "Price not found in kg or ton format."


async def get_material_price(material_name, session):
    """
    获取单个材料的价格信息。
    """
    start_time = asyncio.get_event_loop().time()
    query = f"Latest China domestic price for {material_name} used in AlSi9Mn alloy"
    
    result_template = {
        "name": material_name,
        "price_per_unit": None,
        "unit": "kg",
        "currency": "CNY",
        "source": None,
        "notes": "Search failed or no suitable source found.",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "query_used": query,
    }

    try:
        # 使用 DuckDuckGo 进行搜索以获取数据源 URL
        async with AsyncDDGS() as ddgs:
            search_results = [r async for r in ddgs.text(query, max_results=3)]
        
        if not search_results:
            return result_template

        # 尝试从前几个搜索结果中提取价格
        for result in search_results:
            url = result['href']
            result_template["source"] = url
            try:
                # 设置超时和伪装成浏览器的请求头
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                async with session.get(url, timeout=10, headers=headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'lxml') # 'lxml' is faster
                        page_text = soup.get_text(separator=' ', strip=True)
                        
                        price, notes = parse_price_from_text(page_text, material_name)
                        
                        result_template["notes"] = notes
                        if price is not None:
                            result_template["price_per_unit"] = price
                            # 成功找到价格，停止尝试其他搜索结果
                            break 
                    else:
                        result_template["notes"] = f"Failed to fetch source URL, status code: {response.status}"
            
            except asyncio.TimeoutError:
                result_template["notes"] = "Timeout while fetching source URL."
                continue # 尝试下一个搜索结果
            except Exception as e:
                result_template["notes"] = f"Error processing source URL: {str(e)}"
                continue # 尝试下一个搜索结果

    except Exception as e:
        result_template["notes"] = f"An unexpected error occurred during search: {str(e)}"

    end_time = asyncio.get_event_loop().time()
    logging.debug(f"{material_name} status=200, time={end_time - start_time:.2f}s")
    
    return result_template


async def main():
    """
    主执行函数
    """
    final_data = await find_prices()
    
    print("\n✅ Final structured output:")
    # 使用 json.dumps เพื่อ格式化输出，确保与示例一致
    print(json.dumps(final_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 在Windows上，为asyncio设置正确的事件循环策略
    if asyncio.get_event_loop().is_running():
         # for Jupyter environments
         asyncio.run(main())
    else:
         asyncio.run(main())