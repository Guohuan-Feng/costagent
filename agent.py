# -*- coding: utf-8 -*-
"""
agent.py — LLM-powered process + price reasoning chain
Fully dynamic (no hardcoded alloy composition).
Supports process_finder_tool + price_finder_tool.
"""

import warnings
warnings.filterwarnings("ignore")
from langchain_openai import AzureChatOpenAI
import os, re, json, time, itertools, csv
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Sequence, Optional, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, FunctionMessage, HumanMessage,
    SystemMessage, AIMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from output_parser import LLMCompilerPlanParser, Task
from prompt_template import SYSTEM_PROMPT, JOINER_PROMPT
from tools.process_finder_tool import ProcessFinderTool
from tools.price_finder_tool import PriceFinderTool
from datetime import datetime, timezone, timedelta

# ===========================================================
# === Azure OpenAI 模型初始化 ===
# ===========================================================
load_dotenv()
llm = AzureChatOpenAI(
    deployment_name="gpt-5",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview",
    temperature=1.0
)

# ===========================================================
# === 注册工具 ===
# ===========================================================
process_tool = ProcessFinderTool(llm).as_tool()
price_tool = PriceFinderTool(llm).as_tool()
tools = [process_tool, price_tool]

# ===========================================================
# === 代号提取器（稳定可复用） ===
# ===========================================================
def _extract_alloy_code(text: str) -> str:
    s = (text or "").strip()
    for m in re.finditer(r"\(([^)]+)\)", s):
        inner = m.group(1)
        code = _extract_alloy_code(inner)
        if code:
            return code

    patterns = [
        r"\bEN\s?AC[-\s]?\d{5}\b",
        r"\bADC\d{2}\b",
        r"\b[1-7]\d{3}\b",
        r"\bA3\d{2}\b",
    ]
    for p in patterns:
        m = re.search(p, s, re.I)
        if m: return m.group(0)

    elem = r"(?:[A-Z][a-z]?)(?:\d+(?:\.\d+)?)?"
    m = re.search(rf"\b{elem}(?:[-]{elem}|{elem})+\b", s)
    if m: return m.group(0)
    m = re.search(rf"\b{elem}[-]{elem}\b", s)
    if m: return m.group(0)
    return ""

# ===========================================================
# === 新增：智能分流入口（不修改任何原有逻辑） ===
# ===========================================================

def _llm_is_alloy_name(text: str) -> bool:
    """
    用 GPT-5 智能判断是否为金属合金名（无正则、无硬编码）。
    仅返回 True / False。
    """
    s = (text or "").strip()
    if not s:
        return False
    try:
        check_llm = AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview",
            temperature=1.0
        )
        prompt = f"""
You are a materials expert.
Determine whether the following input refers to a **metal alloy** name or not.

Rules:
- Return ONLY "True" or "False".
- Examples of alloys: AlSi9Mn, CuAl10Ni2, ADC12, EN AC-46000, 6061, Brass, Bronze.
- Examples of non-alloys: Loctite 603, Epoxy 1200, Superglue, Silicone Sealant.
- Be concise. No explanation.

Input: {s}
"""
        resp = check_llm.invoke(prompt)
        ans = (getattr(resp, "content", str(resp)) or "").strip().lower()
        return ans.startswith("t")
    except Exception as e:
        print(f"[WARN] Alloy LLM check failed: {e}")
        return False


def smart_run(user_query: str):
    """
    智能判断用户输入类型，并选择对应工具链运行：
      - 如果 LLM 判断是合金，则保持原流程（process + price）
      - 如果不是合金（如 Loctite 603），仅调用 generic_price_finder_tool
    """
    from tools.generic_price_finder_tool import GenericPriceFinderTool
    from tools.price_finder_tool import PriceFinderTool
    from tools.process_finder_tool import ProcessFinderTool
    from langchain_core.messages import HumanMessage

    print(f"\n[SMART] 🧠 判断材料类型中：{user_query}")
    is_alloy = _llm_is_alloy_name(user_query)
    if is_alloy:
        print("[SMART] ✅ GPT 判断：这是合金 → 走原始链路（process + price）")
        from agent import chain
        result = chain.invoke(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": "smart-run"}}
        )
        return result
    else:
        print("[SMART] ⚙️ GPT 判断：非合金 → 使用 GenericPriceFinderTool 单独执行")
        llm = AzureChatOpenAI(
            deployment_name="gpt-5",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-01-01-preview",
            temperature=1.0
        )
        tool = GenericPriceFinderTool(llm)
        output = tool.run(user_query)  # ✅ 这一行必须加上
        try:
            output_json = json.loads(output)
        except Exception:
            output_json = {"raw_output": output}

        if "final_price_cny_per_g" in output_json:
            output_json["total_cost"] = output_json["final_price_cny_per_g"]
            output_json["unit"] = "CNY/g"

        print("[SMART] 🧾 Generic tool 输出：", output_json)
        return output_json



# ===========================================================
# === LangChain Planner 初始化 ===
# ===========================================================
from prompt_template import SYSTEM_PROMPT
planning_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

def render_tool_for_planning(tool: BaseTool, idx: int) -> str:
    arg_names = list(tool.args.keys()) if hasattr(tool, "args") else []
    sig = ", ".join([f"{a}=..." for a in arg_names]) if arg_names else ""
    return f"{idx}. {tool.name}({sig})\n    - {tool.description.strip()}"

THOUGHT_RE = re.compile(r"(?im)^\s*Thought\s*:\s*(.+?)(?:\n|$)")
def extract_thought(text: str) -> str:
    m = THOUGHT_RE.search(text or "")
    return (m.group(1).strip() if m else "").strip()

def create_planner_components(llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate):
    tool_descriptions = "\n".join(render_tool_for_planning(t, i+1) for i, t in enumerate(tools))
    planner_prompt = base_prompt.partial(
        replan="", num_tools=len(tools)+1, tool_descriptions=tool_descriptions)
    replanner_prompt = base_prompt.partial(
        replan=' - Continue planning using previous results.',
        num_tools=len(tools)+1, tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list): return isinstance(state[-1], SystemMessage)
    def wrap_messages(state: list): return {"messages": state}
    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content += f" - Begin counting at : {next_task}"
        return {"messages": state}

    planner_raw = RunnableBranch(
        (should_replan, wrap_and_get_last_index | replanner_prompt),
        wrap_messages | planner_prompt,
    ) | llm
    planner_tasks = RunnableBranch(
        (should_replan, wrap_and_get_last_index | replanner_prompt),
        wrap_messages | planner_prompt,
    ) | llm | LLMCompilerPlanParser(tools=tools)
    return planner_raw, planner_tasks

planner_raw, planner = create_planner_components(llm, tools, planning_prompt)

# ===========================================================
# === 核心任务执行逻辑 ===
# ===========================================================
def _json_maybe_load(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def _walk_path(obj, path: str):
    if not path: return obj
    cur = obj
    for tok in path.split("."):
        cur = _json_maybe_load(cur)
        if isinstance(cur, list) and tok.isdigit():
            cur = cur[int(tok)]
        elif isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            raise KeyError(f"path '{path}' not found")
    return cur

_REF_ANY = re.compile(r"\$(\d+)(?:\.([A-Za-z0-9_\.]+))?")
def _resolve_inline_refs_in_text(text: str, observations: dict) -> str:
    def _sub(m: re.Match):
        idx = int(m.group(1))
        path = m.group(2) or ""
        base = _json_maybe_load(observations[idx])
        val = _walk_path(base, path)
        return json.dumps(val, ensure_ascii=False)
    return _REF_ANY.sub(_sub, text)

def _resolve_arg(arg, observations):
    if isinstance(arg, str) and "$" in arg:
        try: return _resolve_inline_refs_in_text(arg, observations)
        except Exception: return arg
    if isinstance(arg, list): return [_resolve_arg(a, observations) for a in arg]
    if isinstance(arg, dict): return {k: _resolve_arg(v, observations) for k, v in arg.items()}
    return arg

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results

def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    args = task["args"]
    try:
        resolved_args = _resolve_arg(args, observations)
    except Exception as e:
        return f"ERROR(Failed to resolve args: {repr(e)})"

    if tool_to_use.name == "price_finder":
        ctx_json = observations.get(1)  # ✅ process_finder 的输出永远是 idx=1
        if ctx_json:
            try:
                ctx = json.loads(ctx_json)
                resolved_args["PROCESS_CONTEXT"] = ctx
                print(f"[DEBUG] 🔗 已传递 PROCESS_CONTEXT 给 price_finder")
            except:
                print("[WARN] PROCESS_CONTEXT JSON 解析失败")

    # ✅ 仅在检测到合金代号时打印提示（实际推理由 process_finder 内部完成）
    if tool_to_use.name == "process_finder":
        alloy_text = resolved_args.get("user_request", "")
        code = _extract_alloy_code(alloy_text)
        if code:
            print(f"[DEBUG] ✅ 检测到合金代号 {code}，将在 process_finder 内部进行成分推理。")
        else:
            print(f"[DEBUG] ⚠️ 未检测到任何合金代号，直接进行生产工艺推理。")

    try:
        start = time.time()
        result = tool_to_use.invoke(resolved_args, config)
        end = time.time()
        print(f"[TIME] {tool_to_use.name} 执行耗时: {end - start:.2f} 秒")
        return result
    except Exception as e:
        return f"ERROR(Failed to call {tool_to_use.name}: {repr(e)})"

# ===========================================================
# === 调度执行 ===
# ===========================================================
@as_runnable
def schedule_task(task_inputs, config):
    task, observations = task_inputs["task"], task_inputs["observations"]
    try:
        # 正常执行任务
        observation = _execute_task(task, observations, config)
    except Exception as e:
        # 捕获异常并安全格式化堆栈
        import traceback
        observation = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    # 无论成功或失败都记录结果
    observations[task["idx"]] = observation

def schedule_pending_task(task, observations, retry_after=0.2):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after); continue
        schedule_task.invoke({"task": task, "observations": observations})
        break

@as_runnable
def schedule_tasks(scheduler_input: Dict[str, Any]) -> List[BaseMessage]:
    print("[DEBUG] 进入 schedule_tasks")
    tasks = scheduler_input["tasks"]
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    tool_messages: List[BaseMessage] = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for task in tasks:
            deps = task["dependencies"]
            name = task["tool"].name if not isinstance(task["tool"], str) else task["tool"]
            idx = task["idx"]
            def _submit(_task=task, _name=name, _idx=idx):
                tool_messages.append(AIMessage(content=f"⏳ Start: [{_idx}] {_name}"))
                schedule_task.invoke(dict(task=_task, observations=observations))
                obs = observations[_idx]

                # --- START MODIFICATION: PRINT FUNCTION MESSAGE ---
                print(f"\n===== 📝 FunctionMessage (idx={_idx}, tool={_name}) =====")
                # 尝试漂亮打印 JSON 内容
                try:
                    pretty_content = json.dumps(json.loads(str(obs)), indent=2, ensure_ascii=False)
                    print(pretty_content)
                except:
                    print(str(obs))
                print("====================================================\n")
                # --- END MODIFICATION ---

                tool_messages.append(FunctionMessage(
                    name=_name, content=str(obs),
                    additional_kwargs={"idx": _idx, "args": task["args"]},
                    tool_call_id=_idx))
                tool_messages.append(AIMessage(content=f"✅ Done: [{_idx}] {_name}"))
            if deps and any([dep not in observations for dep in deps]):
                futures.append(executor.submit(schedule_pending_task, task, observations))
            else: _submit()
        wait(futures)
    return tool_messages

# ===========================================================
# === LangGraph 构建 ===
# ===========================================================
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

@as_runnable
def plan_and_schedule(state):
    print("[DEBUG] 进入 plan_and_schedule")
    start_all = time.time()

    messages = state["messages"]

    # ✅ 1. 保存最初的人类输入（只保存一次）
    if "original_user_request" not in state:
        for m in messages:
            if isinstance(m, HumanMessage):
                state["original_user_request"] = m.content
                print(f"[DEBUG] ✅ 记录 original_user_request = {state['original_user_request']}")
                break

    # ✅ 2. 判断是否 Replan 触发
    last_msg = messages[-1]
    is_replan = False
    try:
        repl = json.loads(last_msg.content)
        if repl.get("action") == "replan":
            is_replan = True
            print("[DEBUG] 🔁 检测到 JOINER 触发 Replan，使用 original_user_request 而不是 feedback")
    except:
        pass

    # ✅ 3. 决定这轮工具输入的 user_request 应该是什么：
    if is_replan:
        user_request_for_tools = state["original_user_request"]
    else:
        # 如果不是 Replan，正常使用 messages[-1] 或 HumanMessage 内容
        # 这里保持你的原逻辑，也可以统一用 state["original_user_request"]
        user_request_for_tools = messages[-1].content

    # ✅ 4. 你的用户输入解析逻辑
    user_inputs = parse_user_inputs_from_query(state["original_user_request"])
    print(f"[DEBUG] ✅ 解析的用户输入字段: {user_inputs}")
    state["user_inputs"] = user_inputs

    observations = _get_observations(messages)
    alloy_code = state.get("alloy_code")

    if not alloy_code:
        for m in messages:
            if isinstance(m, HumanMessage):
                code = _extract_alloy_code(str(m.content))
                if code:
                    alloy_code = code
                    print(f"[DEBUG] ✅ 提取到合金代号 {alloy_code}")
                    break

    has_context = (1 in observations)
    task_list = []

    # ✅ 只改这一块的 args，确保不会把 feedback JSON 传给工具
    if not has_context:
        task_list.append({
            "idx": 1,
            "tool": process_tool,
            "args": {"user_request": user_request_for_tools},
            "dependencies": []
        })
        print("[DEBUG] 📌 需要工艺 → 添加 process_finder")

    task_list.append({
        "idx": 2,
        "tool": price_tool,
        "args": {"user_request": user_request_for_tools},
        "dependencies": [1] if not has_context else []
    })
    print("[DEBUG] 📌 添加 price_finder")

    # ✅ 保留你的 planner 思考记录逻辑
    raw = planner_raw.invoke(messages)
    raw_content = getattr(raw, "content", "")
    
    # --- START MODIFICATION: PRINT PLANNER AIMESSAGE ---
    print("\n===== 🧠 Planner AIMessage (Thought/Action) =====")
    print(raw_content)
    print("================================================\n")
    # --- END MODIFICATION ---

    thought_text = extract_thought(raw_content)
    if thought_text:
        messages = messages + [AIMessage(content=f"Thought: {thought_text}")]

    print("[DEBUG] ✅ 动态任务列表: ", task_list)

    start_sched = time.time()
    tool_messages = schedule_tasks.invoke({"messages": messages, "tasks": task_list})
    end_sched = time.time()
    print(f"[TIME] schedule_tasks.invoke 耗时: {end_sched - start_sched:.2f} 秒")

    total = time.time() - start_all
    print(f"[TIME] plan_and_schedule 总耗时: {total:.2f} 秒")

    return {"messages": messages + tool_messages, "alloy_code": alloy_code}



class FinalResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    json_output: Optional[Dict[str, Any]] = Field(None) # <--- 修改为 Optional[Dict] = Field(None)

class Replan(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    feedback: Optional[str] = Field(None, validation_alias=AliasChoices("feedback", "Feedback")) # <--- 修改为 Optional[str] = Field(None)

class JoinOutputs(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    thought: str = Field(validation_alias=AliasChoices("thought", "Thought"))
    action: Union[FinalResponse, Replan]

joiner_prompt = ChatPromptTemplate.from_template(JOINER_PROMPT).partial(examples="")
joiner_llm = AzureChatOpenAI(
    deployment_name="gpt-5",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview",
    temperature=1.0
)
runnable = joiner_prompt | joiner_llm.with_structured_output(JoinOutputs, method="function_calling")
JOIN_SENTINEL = "__JOIN_DONE__"

def make_final_ai_md(text: str) -> str:
    return f"{text}\n\n"

def build_join_parser_with_context():
    def _parse_joiner_output(decision):
        print("[DEBUG] joiner 输出 decision:", decision)

        action = decision.action

        # ✅ 正确解析 json_output 并确认包含 total_cost
        if hasattr(action, "json_output"):
            result = action.json_output
            if isinstance(result, dict) and "total_cost" in result:
                
                # --- START MODIFICATION: PRINT JOINER AIMESSAGE ---
                final_json_content = json.dumps(result, indent=2, ensure_ascii=False)
                print("\n===== 💰 Joiner AIMessage (Final Cost JSON) =====")
                print(final_json_content)
                print("==================================================\n")
                # --- END MODIFICATION ---
                
                msgs = [
                    AIMessage(content=final_json_content),
                    SystemMessage(content=JOIN_SENTINEL),
                ]
                return {"messages": msgs}

        # ❌ 如果不是 json_output 或缺失 total_cost → Replan
        feedback = getattr(action, "feedback", "JOINER missing json_output.total_cost")
        msgs = [
            SystemMessage(content=json.dumps(
                {"action": "replan", "feedback": feedback},
                ensure_ascii=False
            ))
        ]
        return {"messages": msgs}

    return _parse_joiner_output




def select_recent_messages(state) -> dict:
    messages = state["messages"]; selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage): break
    return {"messages": selected[::-1]}

joiner = select_recent_messages | runnable | build_join_parser_with_context()
class State(TypedDict): 
    messages: List[BaseMessage]
    alloy_code: Optional[str]
checkpointer = None
graph_builder = StateGraph(State)
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge(START, "plan_and_schedule")
graph_builder.add_edge("plan_and_schedule", "join")

def should_continue(state):
    for m in state["messages"]:
        if isinstance(m, SystemMessage) and m.content == JOIN_SENTINEL:
            return END
    return "plan_and_schedule"

graph_builder.add_conditional_edges("join", should_continue)
chain = graph_builder.compile(checkpointer=checkpointer)
print("🟢 Chain execution finished.")

import os
import re
from langchain_core.messages import AIMessage


# --- agent.py 修正版：支持 triple-quoted long_output 解析 ---
import os, re, json
from datetime import datetime

def _extract_alloy_code_from_messages(messages) -> str:
    try:
        for msg in messages[::-1]:
            text = str(getattr(msg, "content", "") or "")
            code = _extract_alloy_code(text)
            if code:
                return code
        return "Alloy"
    except:
        return "Alloy"

def extract_long_output_from_text(text: str) -> str:

    # 兼容 llm 生成的三引号格式：
    # long_output=""" ... """
    # 并剥离外层代码块 ```...```

    # 先找 long_output=""" 
    pat = re.compile(r'long_output\s*=\s*"""\n?(.*?)"""', re.DOTALL)
    m = pat.search(text)
    if not m:
        return None
    content = m.group(1).strip()

    # 去掉 ``` 包裹
    if content.startswith("```"):
        content = content.lstrip("`")
        # 去掉末尾对应的 ```
        content = content.rstrip("`")
        # 防止语言标记如 ```markdown
        content = re.sub(r'^.*?\n', '', content, count=1)

    return content.strip()

import uuid, os, json
from datetime import datetime

def save_json_report(state):
    """
    自动保存长/短 JSON，并上传到 Azure Blob Storage (rates/longjson, rates/shortjson)
    """
    messages = state.get("messages", [])
    # ✅ 尝试读取 state 中的 user_inputs
    user_inputs = state.get("user_inputs", {})

    # ✅ 如果为空，则自动从用户 query 重新解析
    if not user_inputs:
        for m in messages:
            if isinstance(m, HumanMessage):
                user_inputs = parse_user_inputs_from_query(m.content)
                break

    # ✅ 保存回 state，确保后续可以用
    state["user_inputs"] = user_inputs     
    final_json = None

    # 1️⃣ 找到包含 total_cost 的最终 JSON 输出
    for m in messages[::-1]:
        try:
            parsed = json.loads(m.content)
            print("[DEBUG] 🧩 尝试解析 JSON：", parsed)
            if "total_cost" in parsed:
                final_json = parsed
                break
        except Exception as e:
            print(f"[WARN] 无法解析消息: {m.content[:80]}... ({e})")

    if not final_json:
        print("[WARN] 未找到最终 JSON 输出")
        return

    # 2️⃣ 添加北京时间时间戳（紧凑格式）
    sg_tz = timezone(timedelta(hours=8))
    now = datetime.now(sg_tz)
    timestamp = datetime.now(sg_tz).strftime("%Y%m%d_%H%M%S")
    final_json["timestamp"] = timestamp

    # 3️⃣ 获取合金名（默认 UnknownAlloy）
    alloy_name = final_json.get("alloy_code", "UnknownAlloy")

    # 4️⃣ 本地保存路径（改为安全路径避免 OneDrive 冲突）
    base_dir = os.path.join(os.path.expanduser("~"), "Documents", "costagent_output")
    long_dir = os.path.join(base_dir, "Detailed_Material_Rate")
    short_dir = os.path.join(base_dir, "Brief_Material_Rate")
    os.makedirs(long_dir, exist_ok=True)
    os.makedirs(short_dir, exist_ok=True)

    # 5️⃣ 文件命名：合金名 + 紧凑时间戳
    filename = f"{alloy_name}_{timestamp}.json"
    long_path = os.path.join(long_dir, filename)
    short_path = os.path.join(short_dir, filename)

    # 6️⃣ 写入长短 JSON 文件
    with open(long_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    short_json = {
        "timestamp": final_json.get("timestamp"),
        "alloy_code": final_json.get("alloy_code"),
        "unit": final_json.get("unit"),
        "total_cost": final_json.get("total_cost"),
    }
    with open(short_path, "w", encoding="utf-8") as f:
        json.dump(short_json, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"[INFO] ✅ 已生成长 JSON: {long_path}")
    print(f"[INFO] ✅ 已生成短 JSON: {short_path}")

    # ----------------------------------------
    # 删除了 Azure Blob Storage 上传相关代码
    # ----------------------------------------
    
    # ============== ✅ 保存到 CSV ==============
    try:
        # 提取 state 中的字段
        user_inputs = state.get("user_inputs", {})

        # 必须字段，没有输入的留空
        Location = user_inputs.get("Location", "")
        supplier_code = user_inputs.get("supplier_code", "")
        part_number = user_inputs.get("part_number", "")
        sub_process_step = user_inputs.get("sub_process_step", "")
        process_type = user_inputs.get("process_type", "")

        # 从 JSON 中获取材料名、时间、成本
        material_name = final_json.get("alloy_code") or final_json.get("material", "Unknown")
        Low = float(final_json.get("total_cost", 0))
        High = round(Low + 2, 2)
        Unit = "/kg"
        valid_time = final_json.get("last_updated", final_json.get("timestamp", ""))
        csv_time = now.strftime("%#m/%#d/%Y")  # 例如 11/3/2025
        source = "web"

        # 组织输出行
        row = {
            "Location": Location,
            "supplier_code": supplier_code,
            "part_number": part_number,
            "sub_process_step": sub_process_step,
            "material_name": material_name,
            "process_type": process_type,
            "Low": Low,
            "High": High,
            "Unit": Unit,
            "valid_time": csv_time,
            "source": source
        }
        # ✅ 创建 csv 文件夹
        csv_dir = os.path.join(base_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, "cost_records.csv")

        # ✅ 写入 CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"✅ CSV 已写入: {csv_path}")
    except Exception as e:
        print(f"❌ CSV 保存失败: {e}")
    print("🏁 所有文件上传完成。")


import re

def parse_user_inputs_from_query(query: str) -> dict:
    """
    ✅ 从 query 中提取 key=value 对 (支持逗号、空格、多字段)
    ✅ 返回字典，例如：
      {"Location": "Ningbo, Zhejiang", "supplier_code": "97036203", ...}
    """
    result = {}
    pairs = re.findall(r'(\w+)\s*=\s*([^;]+)', query)
    for key, val in pairs:
        result[key.strip()] = val.strip()
    return result

