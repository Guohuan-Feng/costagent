from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv()

# 设置代理
os.environ["HTTP_PROXY"] = os.getenv("PROXY_URL")
os.environ["HTTPS_PROXY"] = os.getenv("PROXY_URL")

# 初始化 LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-5",  # Azure 上你的部署名称
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview",
    temperature=1.0
)

# 测试模型调用
try:
    response = llm.invoke("Say hello from Azure GPT-5.")
    print("✅ 模型响应成功：", response.content)
except Exception as e:
    print("❌ 模型调用失败：", e)
