## Description
This project is an AI agent designed to answer questions based on loaded documentation using RAG (Retrieval-Augmented Generation) architecture.  
It includes modules for document preprocessing, text vectorization (embedding), clustering, storage, and pipeline orchestration.


## Installation

1. Clone the repository and navigate to the project folder  
   ```bash
   git clone <repo-url>
   cd RAG-system
   ```

2. Create and activate a Python virtual environment  
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables based on example  
   ```bash   
   # Edit .env with your API keys and settings
   ```

## Project Structure

```
ai-agent-documentation/
│
├── .gitignore
├── .env
├── README.md
├── requirements.txt
├── main.py
│
├── src/
│   ├── models.py           # Data models (ProcessorResult, EmbedderResult, etc.)
│   ├── utils.py            # Utilities (env loading, get_prompt_by_id, logging)
│   │
│   ├── preprocessing/      # Preprocessor & text cleaning strategies
│   ├── embeddings/         # Embedder & embedding implementations
│   ├── storage/            # Storage & storage implementations
│   ├── pipeline/           # Pipeline orchestration and stages
│   └── agent/              # Main AI agent interface
│
├── prompts/                # Prompts JSON and templates
│
├── config/                 # Configuration and logging setup
│
├── tests/                  # Unit tests for components
│
└── data/                   # Data folders (raw, processed, embeddings)
```

## Running the Project

```bash
python main.py
```

## Development and Git Workflow

- Create feature branches:
  ```bash
  git checkout -b feature/preprocessor
  ```
- After implementing tasks, make a Pull Request to `main`/`master`
- Reference Jira issues and provide summary in PR description

## Testing

```bash
pytest tests/ --cov=src
```
