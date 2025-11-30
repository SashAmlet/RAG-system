# RAG System with LangChain & Ollama

## Description
AI-powered question-answering system built with **RAG (Retrieval-Augmented Generation)** architecture. The system uses **LangChain** and **LangGraph** for agent orchestration, **Ollama** for local LLM inference, **FAISS** for vector storage, and **Sentence Transformers** for embeddings.

### Key Features
- 🤖 **LangGraph-based Agent** - Modular RAG workflow with retrieve → prompt → generate pipeline
- 🏠 **Local LLM** - Runs on Ollama (Qwen2.5:7b) - no API costs
- 🌐 **Web UI** - Interactive chat interface via LangServe Playground
- 📚 **Document Processing** - Supports PDF and TXT files with advanced preprocessing
- 🔍 **Semantic Search** - FAISS vector database with similarity search
- 💬 **CLI Mode** - Interactive terminal interface for quick queries

## Prerequisites

1. **Install Ollama** and download the model:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull qwen2.5:7b
   ```

2. **Python 3.8+** with virtual environment support

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd RAG-system
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional):
   ```bash
   # Edit .env file if needed
   # Default values work out of the box
   ```

## Quick Start

### 1. Index Your Documents

Place PDF or TXT files in `data/raw/`, then run:

```bash
python main.py --mode index
```

This will:
- Process and clean documents
- Generate embeddings using Sentence Transformers
- Store vectors in FAISS index at `data/indexes/knowledge_base`

### 2. Start Ollama server

```bash
ollama serve
```

### 3. Start the Web Interface

```bash
python server.py
```

Open your browser at: **http://localhost:8000/rag/playground/**

You'll see an interactive chat interface where you can ask questions about your documents.

### 4. CLI Mode (Alternative)

For terminal-based interaction:

```bash
python main.py --mode interactive
```

Or single query mode:

```bash
python main.py --mode query --question "Що таке мікросервіси?"
```

## Project Structure

```
RAG-system/
│
├── server.py              # LangServe web server entry point
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
│
├── src/
│   ├── agent/             # LangGraph agent & LLM clients
│   │   ├── agent.py       # AIAgent with LangGraph workflow
│   │   ├── llm_client.py  # Ollama & Perplexity clients
│   │   ├── retriever.py   # Document retrieval logic
│   │   └── prompt_builder.py
│   │
│   ├── preprocessing/     # Document processing pipeline
│   ├── embeddings/        # Sentence Transformers embedder
│   ├── storage/           # FAISS vector storage
│   └── models.py          # Data models
│
├── data/
│   ├── raw/               # Place your documents here
│   └── indexes/           # Generated FAISS indexes
│
├── prompts/               # System prompts
├── config/                # Logging configuration
└── tests/                 # Unit tests
```

## Architecture

The system uses **LangGraph** to orchestrate the RAG workflow:

```
User Query → Retrieve (FAISS) → Build Prompt → Generate (Ollama) → Response
```

### Components

1. **Retriever** - Finds relevant document chunks using semantic search
2. **Prompt Builder** - Constructs context-aware prompts
3. **LLM Client** - Generates answers via Ollama (local) or Perplexity API
4. **Agent** - Orchestrates the workflow using LangGraph

## Configuration

Edit `.env` to customize:

```bash
# LLM Settings
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=800

# Embeddings
EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval
TOP_K=5
MIN_SIMILARITY=0.3

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

## API Documentation

When `server.py` is running, visit:
- **Playground**: http://localhost:8000/rag/playground/
- **API Docs**: http://localhost:8000/docs

## Development

### Running Tests

```bash
pytest tests/ --cov=src
```

### Git Workflow

```bash
git checkout -b feature/your-feature
# Make changes
git commit -m "Description"
git push origin feature/your-feature
```

## Technologies

- **LangChain** & **LangGraph** - Agent framework
- **Ollama** - Local LLM runtime
- **FAISS** - Vector database
- **Sentence Transformers** - Embeddings
- **FastAPI** - Web server
- **LangServe** - LangChain deployment

## Troubleshooting

### "Index not found"
Run `python main.py --mode index` first to create the index.

### "Ollama connection error"
Make sure Ollama is running: `ollama serve`

### Empty responses in web UI
Check server logs for `DEBUG:` messages. Restart server after code changes.

## License

MIT
