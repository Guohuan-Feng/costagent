# ==========================================
# process_finder_tool.py (LLM only — no hardcoding)
# ==========================================

from __future__ import annotations
import json, re, os
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class ProcessFinderArgs(BaseModel):
    user_request: str = Field(
        ...,
        description="User's natural language question about a material or alloy production process."
    )

class ProcessFinderTool:
    """
    Dynamically generates structured knowledge of alloy/material production processes in JSON format.
    LLM-based composition inference (no hardcoding).
    Output automatically compressed (<400 chars).
    """

    def __init__(self, llm: Optional[BaseChatModel] = None):
        self.name = "process_finder"
        self.description = (
            "Infer composition (full element names) and analyze material/alloy production process. "
            "Outputs compact JSON (<400 chars) with composition, raw materials, process stages, and energy types."
        )
        self.llm = llm or AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview",
            temperature=1.0
        )

    def _infer_composition(self, material_name: str) -> dict:
        """
        ✅ Fully LLM-based composition inference
        ✅ No regex rules to derive percentages
        ✅ No element symbol mapping table
        ✅ Based entirely on metallurgical naming conventions
        """
        prompt = f"""
You are a senior metallurgical engineer.

Infer the **mass fraction composition** (sum = 1.00)
for the alloy or material: "{material_name}"

Rules:
- Use metallurgical knowledge (e.g., CuAl10Ni2 → Al≈10%, Ni≈2%, remainder Cu)
- Convert element symbols to full English names
- Fractions must be decimal (0.88 not 88)
- Sum must equal exactly 1.00
- Output pure JSON only:
  {{"composition": {{"Element": fraction, ...}}}}
- No explanation. No markdown. No comments.
"""
        response = self.llm.invoke(prompt)
        text = response.content.strip()

        # Extract pure JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid composition JSON: {text}")
        data = json.loads(match.group(0))
        return data["composition"]

    def run(self, user_request: str) -> str:
        """Main entry: infer composition + production process in compact JSON."""
        try:
            # ✅ Use full query as material name (no parsing or hardcoding)
            material_name = user_request.strip()

            # ① Composition inference
            comp = self._infer_composition(material_name)

            # ② Manufacturing process inference
            prompt = f"""
You are a senior metallurgical process engineer.
Infer a **compact** JSON description (<400 chars)
for producing material: "{material_name}"

Requirements:
- Output: compact pure JSON
- Keys:
  {{
    "composition": {json.dumps(comp, ensure_ascii=False)},
    "raw_materials": ["<inputs>"],
    "process_stages": ["<stages>"],
    "energy": ["<sources>"]
  }}
"""
            resp = self.llm.invoke(prompt)
            text = getattr(resp, "content", str(resp)).strip()
            text = text.replace("```json", "").replace("```", "").strip()

            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]

            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

        except Exception as e:
            return json.dumps({"error": f"ProcessFinder failed: {e}"}, ensure_ascii=False)

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=ProcessFinderArgs
        )

if __name__ == "__main__":
    tool = ProcessFinderTool()
    example = "Explain the production process of AlSi9Mn alloy."
    print(tool.run(example))
