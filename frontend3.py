import streamlit as st
import tempfile, os
from backend import chatbot, tools
from rag_tool import load_documents, clear_documents, has_documents
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id, "New Chat")
    st.session_state['message_history'] = []

def add_thread(thread_id, title="New Chat"):
    if thread_id not in [t["id"] for t in st.session_state['chat_threads']]:
        st.session_state['chat_threads'].append({"id": thread_id, "title": title})

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ── Session state init ────────────────────────────────────────────────────────
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []
if 'thread_id' not in st.session_state:
    first_thread = generate_thread_id()
    st.session_state['thread_id'] = first_thread
    st.session_state['chat_threads'] = [{"id": first_thread, "title": "New Chat"}]
if "message_history" not in st.session_state:
    st.session_state['message_history'] = []
if "uploaded_doc_names" not in st.session_state:
    st.session_state["uploaded_doc_names"] = []

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🤖 AI Chatbot")
st.sidebar.button("➕ New Chat", on_click=reset_chat)

# ── RAG: Document Upload ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload (RAG)")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="doc_uploader",
)

if st.sidebar.button("📥 Index Documents"):
    if uploaded_files:
        with st.spinner("Indexing documents..."):
            tmp_paths = []
            for f in uploaded_files:
                suffix = os.path.splitext(f.name)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_paths.append(tmp.name)
            count = load_documents(tmp_paths)
            # Clean up temp files
            for p in tmp_paths:
                os.unlink(p)
            st.session_state["uploaded_doc_names"] = [f.name for f in uploaded_files]
        st.sidebar.success(f"✅ Indexed {count} chunks from {len(uploaded_files)} file(s)")
    else:
        st.sidebar.warning("Please upload at least one file first.")

if st.sidebar.button("🗑️ Clear Documents"):
    clear_documents()
    st.session_state["uploaded_doc_names"] = []
    st.sidebar.success("Documents cleared.")

if st.session_state["uploaded_doc_names"]:
    st.sidebar.markdown("**Indexed files:**")
    for name in st.session_state["uploaded_doc_names"]:
        st.sidebar.markdown(f"- 📄 `{name}`")
elif not has_documents():
    st.sidebar.info("No documents indexed yet.")

# ── Tools info ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Tools loaded:** {len(tools)}")
with st.sidebar.expander("Available Tools"):
    for tool in tools:
        st.markdown(f"- `{tool.name}`")

# ── Chat History ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("💬 Chat History")
for thread in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread["title"], key=str(thread["id"])):
        st.session_state['thread_id'] = thread["id"]
        messages = load_conversation(thread["id"])
        temp_msg = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_msg.append({"role": "user", "content": msg.content, "tool_calls": []})
            elif isinstance(msg, AIMessage):
                temp_msg.append({"role": "assistant", "content": msg.content, "tool_calls": []})
        st.session_state['message_history'] = temp_msg


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("🤖 AI Chatbot")
st.caption("Math tools + RAG document search powered by Groq & FastMCP")

# Render existing chat history
for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("tool_calls"):
            with st.expander("🔧 Tool calls", expanded=False):
                for tc in message["tool_calls"]:
                    st.markdown(f"**Tool:** `{tc['name']}`")
                    st.markdown(f"**Args:** `{tc['args']}`")
                    st.markdown(f"**Result:** `{tc['result']}`")

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a math question or query your documents...")

if user_input:
    for thread in st.session_state["chat_threads"]:
        if (
            thread["id"] == st.session_state["thread_id"]
            and thread["title"] == "New Chat"
        ):
            thread["title"] = user_input[:30]
            break

    st.session_state['message_history'].append({"role": "user", "content": user_input, "tool_calls": []})
    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        ai_text = ""
        tool_calls_info = []
        pending_tool_calls = {}

        stream = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        )

        text_placeholder = st.empty()

        for message_chunk, metadata in stream:
            if isinstance(message_chunk, AIMessage):
                if message_chunk.tool_calls:
                    for tc in message_chunk.tool_calls:
                        pending_tool_calls[tc["id"]] = {
                            "name": tc["name"],
                            "args": tc["args"],
                        }
                if message_chunk.content:
                    ai_text += message_chunk.content
                    text_placeholder.markdown(ai_text)

            elif isinstance(message_chunk, ToolMessage):
                call_id = message_chunk.tool_call_id
                if call_id in pending_tool_calls:
                    tc = pending_tool_calls.pop(call_id)
                    tool_calls_info.append({
                        "name": tc["name"],
                        "args": tc["args"],
                        "result": message_chunk.content,
                    })

        # ── Source badge ──────────────────────────────────────────────────
        rag_calls  = [tc for tc in tool_calls_info if tc["name"] == "rag_search"]
        math_calls = [tc for tc in tool_calls_info if tc["name"] != "rag_search"]

        if rag_calls:
            st.info("📄 This response was generated using your uploaded documents.")
        if math_calls:
            st.info("🔢 This response used math tools.")
        if not tool_calls_info:
            st.caption("💬 This response came from the LLM's own knowledge.")

        if tool_calls_info:
            with st.expander("🔧 Tool calls", expanded=False):
                for tc in tool_calls_info:
                    icon = "📄" if tc["name"] == "rag_search" else "🔢"
                    st.markdown(f"**{icon} Tool:** `{tc['name']}`")
                    st.markdown(f"**Args:** `{tc['args']}`")
                    if tc["name"] == "rag_search":
                        with st.expander("📖 Retrieved chunks", expanded=False):
                            st.markdown(tc["result"])
                    else:
                        st.markdown(f"**Result:** `{tc['result']}`")
                    st.markdown("---")

    st.session_state['message_history'].append({
        "role": "assistant",
        "content": ai_text,
        "tool_calls": tool_calls_info,
    })
