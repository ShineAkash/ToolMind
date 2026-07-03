# 🧠 ToolMind

> An AI chatbot that *thinks* about which tool to use — combining **RAG**, **MCP**, and **LangGraph** into a single multi-thread conversational agent.

ToolMind is a portfolio-grade showcase of modern agentic AI architecture. A user asks a question, and the agent autonomously decides whether to search uploaded documents, call a remote math tool over MCP, or answer from its own knowledge — all orchestrated through a LangGraph state machine with persistent multi-thread memory.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.32+-FF4B4B.svg)
![LangGraph](https://img.shields.io/badge/langgraph-0.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## ✨ Features

- 🧠 **LangGraph agent** with ReAct-style tool routing and stateful checkpointing
- 🔌 **MCP integration** over streamable-HTTP for remote tool discovery
- 📄 **RAG pipeline** — PDF/TXT ingestion, semantic chunking, FAISS vector search
- 🔧 **Multi-tool selection** — RAG, MCP math tools, and LLM knowledge
- 💬 **Multi-thread chat history** with persistent memory per thread
- ⚡ **Real-time streaming** of tokens and tool-call events
- 🛡️ **Schema-driven argument coercion** for reliable LLM tool calls
- 🎨 **Streamlit UI** with document upload, tool-call transparency, and chat sidebar

---

## 🏗️ Architecture

```
                     ┌──────────────────────┐
                     │  User (Streamlit UI) │
                     └──────────┬───────────┘
                                │ user message
                                ▼
                     ┌──────────────────────┐
                     │   LangGraph Agent    │
                     │   (StateGraph)       │
                     └──────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         ┌──────────┐    ┌──────────┐    ┌──────────┐
         │   RAG    │    │   MCP    │    │   LLM    │
         │  Tool    │    │  Math    │    │   Own    │
         │          │    │  Tools   │    │ Knowledge│
         └────┬─────┘    └────┬─────┘    └──────────┘
              │               │
              ▼               ▼
        ┌──────────┐   ┌─────────────┐
        │  FAISS   │   │ Remote MCP  │
        │  + HF    │   │   Server    │
        │ Embed    │   │  (HTTP)     │
        └──────────┘   └─────────────┘
```

**Flow:**
1. User sends a message via the Streamlit UI
2. LangGraph routes it through a `chat` node powered by Groq's LLM
3. The LLM decides whether to call a tool (`tools_condition` edge)
4. If a tool is called, the `tools` node executes it and loops back
5. The final response is streamed token-by-token back to the UI
6. All state is checkpointed per `thread_id` for persistent memory

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq (`meta-llama/llama-4-scout-17b-16e-instruct`) |
| **Agent Framework** | LangGraph (StateGraph + InMemorySaver) |
| **Tool Protocol** | Model Context Protocol (MCP) via `langchain-mcp-adapters` |
| **Vector Store** | FAISS (in-memory) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Document Loaders** | PyPDFLoader, TextLoader |
| **Chunking** | RecursiveCharacterTextSplitter (chunk=500, overlap=50) |
| **UI** | Streamlit |
| **Config** | python-dotenv |

---

## 📂 Project Structure

```
toolmind/
├── backend.py          # LangGraph agent + MCP client
├── frontend3.py        # Streamlit UI
├── rag_tool.py         # RAG pipeline (load, chunk, embed, search)
├── requirements.txt    # Python dependencies
├── .env                # Secrets (not committed)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/toolmind.git
cd toolmind
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
MCP_MATH_URL=https://your-mcp-server-url/mcp
```

> **Get a Groq API key:** [https://console.groq.com/keys](https://console.groq.com/keys) (free tier available)

### 5. Run the app

```bash
streamlit run frontend3.py
```

The app will open at `http://localhost:8501`.

---

## 💡 Usage Examples

### Math via MCP
> **You:** What's 145 multiplied by 23?
>
> **ToolMind:** *calls MCP `multiply` tool* → **3,335**

### Document RAG
> **You:** *uploads `company-handbook.pdf`* → "What is the leave policy?"
>
> **ToolMind:** *calls `rag_search` tool* → retrieves relevant chunks → answers with sources

### Pure LLM knowledge
> **You:** Explain quantum entanglement in simple terms.
>
> **ToolMind:** *no tool calls* → answers directly from the LLM's training data

The UI clearly indicates which source powered each response:
- 📄 **RAG** — "This response was generated using your uploaded documents"
- 🔢 **MCP** — "This response used math tools"
- 💬 **LLM** — "This response came from the LLM's own knowledge"

---

## 🔬 Technical Highlights

### LangGraph Agent Loop
A `StateGraph` with two nodes (`chat`, `tools`) connected by a conditional edge. The LLM decides whether to call a tool; if it does, the tool runs and the result loops back to the LLM — a ReAct-style reasoning loop.

### Multi-Thread Memory
`InMemorySaver` checkpointer stores full conversation state per `thread_id`. The sidebar lets you switch between independent chat threads, each with its own history.

### MCP over HTTP
The math tools live on a **remote MCP server**, not in this codebase. This decouples tool implementation from the agent — you can add new MCP servers without touching the agent.

### Schema-Driven Argument Coercion
LLMs sometimes emit `"5"` instead of `5` for numeric arguments. A custom `coerce_tool_args` function reads each tool's JSON schema and coerces types at runtime — defensive engineering for unreliable LLM outputs.

### Async MCP + Sync LangChain Interop
MCP tools are async; `rag_search` is sync. A dedicated event loop bridges the two — a real-world problem when mixing tool types.

### Streaming Tool Events
`chatbot.stream(..., stream_mode="messages")` streams both LLM tokens and tool-call results in real time, so the UI shows progress as the agent reasons.

---

## ☁️ Deployment

### Streamlit Cloud (recommended)

1. Push the repo to GitHub (do **not** commit `.env` or `venv/`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Point to your repo, set main file to `frontend3.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_key"
   MCP_MATH_URL = "your_mcp_url"
   ```
5. Deploy 🚀

> **Note:** Streamlit Cloud's free tier has ~1 GB RAM. The `all-MiniLM-L6-v2` embedding model (~80 MB) loads fine, but the first message will be slower as the model warms up.

---

## 🛡️ Tech Concepts Demonstrated

- **Agentic AI** — ReAct loop with autonomous tool selection
- **Function Calling** — structured tool-use via `llm.bind_tools()`
- **RAG** — document ingestion → chunking → embeddings → vector search → grounded generation
- **MCP** — Model Context Protocol over streamable-HTTP
- **LangGraph** — stateful multi-node graph with checkpointing
- **Streaming** — real-time token + tool-event output
- **Semantic Search** — meaning-based retrieval via embeddings
- **Document Chunking** — recursive text splitting with overlap
- **Stateful Memory** — multi-thread conversation persistence
- **Schema-driven Argument Coercion** — defensive LLM output handling
- **Async/Sync Interop** — bridging event-loop-based MCP and sync LangChain tools
- **Secrets Management** — `.env` with fail-fast validation

---

## 📜 License

MIT License — feel free to use this as a learning reference.

---

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) — agent orchestration
- [Model Context Protocol](https://modelcontextprotocol.io/) — tool protocol
- [Groq](https://groq.com/) — fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — vector search
- [HuggingFace](https://huggingface.co/) — embeddings
- [Streamlit](https://streamlit.io/) — UI framework
