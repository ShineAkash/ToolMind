import asyncio
import os
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.prebuilt import tools_condition
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from rag_tool import rag_search

load_dotenv()

MCP_MATH_URL = os.getenv("MCP_MATH_URL")
if not MCP_MATH_URL:
    raise RuntimeError(
        "MCP_MATH_URL is not set. Add it to your .env file (local) or "
        "Streamlit Cloud secrets (deployed)."
    )

# Default model — change via GROQ_MODEL in .env if a model is deprecated/removed
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

SERVERS = {
    "math-mcp": {
        "transport": "streamable-http",
        "url": MCP_MATH_URL,
    }
}

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant. "
    "You have access to math tools — use them for any math-related questions. "
    "You also have a rag_search tool — use it whenever the user asks about "
    "information that may be in their uploaded documents. "
    "Only call tools that are explicitly provided to you. "
    "Always pass numeric arguments as actual numbers, never as strings."
))


def coerce_tool_args(tool, args: dict) -> dict:
    """Coerce tool arguments to correct types based on the tool's JSON schema.

    Handles two schema shapes:
    - LangChain tools: ``tool.args_schema`` is a Pydantic class with ``model_json_schema()``.
    - MCP tools: ``tool.args_schema`` is absent; the schema is at ``tool.inputSchema`` (dict).
    """
    properties: dict = {}

    pydantic_schema = getattr(tool, "args_schema", None)
    if pydantic_schema is not None:
        try:
            properties = pydantic_schema.model_json_schema().get("properties", {}) or {}
        except Exception:
            properties = {}

    # MCP tools expose a raw JSON schema dict at ``inputSchema``
    if not properties:
        mcp_schema = getattr(tool, "inputSchema", None)
        if isinstance(mcp_schema, dict):
            properties = mcp_schema.get("properties", {}) or {}

    if not properties:
        return args

    coerced = {}
    for key, value in args.items():
        prop = properties.get(key, {})
        expected_type = prop.get("type")
        try:
            if expected_type == "integer":
                coerced[key] = int(float(value))
            elif expected_type == "number":
                coerced[key] = float(value)
            elif expected_type == "boolean":
                coerced[key] = bool(value)
            else:
                coerced[key] = value
        except (ValueError, TypeError):
            coerced[key] = value
    return coerced


def build_graph(all_tools: list):
    llm = ChatGroq(model=GROQ_MODEL)
    llm_with_tools = llm.bind_tools(all_tools)

    # named_tools: MCP tools are async, rag_search is sync
    named_tools = {tool.name: tool for tool in all_tools}

    def chat_node(state: MessagesState):
        messages = [SYSTEM_PROMPT] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def tool_node_fn(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        tool_results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id   = tool_call["id"]

            if tool_name not in named_tools:
                tool_results.append(ToolMessage(
                    content=f"Error: tool '{tool_name}' not available.",
                    tool_call_id=tool_id,
                ))
                continue

            the_tool = named_tools[tool_name]
            coerced  = coerce_tool_args(the_tool, tool_args)

            # rag_search is a sync LangChain tool
            if tool_name == "rag_search":
                result = the_tool.invoke(coerced)
            else:
                # MCP tools are async — run via dedicated event loop
                result = _loop.run_until_complete(the_tool.ainvoke(coerced))
                # Unwrap MCP content blocks if needed
                if isinstance(result, list):
                    result = " ".join(
                        item["text"] if isinstance(item, dict) and "text" in item
                        else str(item)
                        for item in result
                    )

            tool_results.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
            ))

        return {"messages": tool_results}

    checkpointer = InMemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node_fn)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=checkpointer)


# ── Initialize MCP tools + RAG tool at import time ───────────────────────────
def _init():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    client = MultiServerMCPClient(SERVERS)
    mcp_tools = loop.run_until_complete(client.get_tools())
    all_tools = mcp_tools + [rag_search]   # ← add RAG tool here
    return all_tools, loop


tools, _loop = _init()
chatbot = build_graph(tools)
