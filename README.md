## Description
This project is an AI agent designed to answer questions based on loaded documentation using RAG (Retrieval-Augmented Generation) architecture.  
It includes modules for document preprocessing, text vectorization (embedding), clustering, storage, and pipeline orchestration.

## Upcoming Sprint Tasks

### 1. Implement the Preprocessor
- The `Preprocessor` class should:
  - Accept input files (.pdf, .doc, .docx)
  - Extract text while cleaning unnecessary content (images, links, etc.)
  - Handle special cases such as tables and hyphenations
  - Return a `ProcessorResult` model containing cleaned text and metadata
- Architecture:
  - Use the Strategy pattern for text cleaning strategies (`TextCleaner`)
  - Use the Factory pattern to easily add support for new file types without modifying main code

### 2. Implement the Embedder
- The `Embedder` class should:
  - Take in `ProcessorResult` from Preprocessor
  - Convert cleaned text to vector representations using embedding techniques
  - Return an `EmbedderResult` model containing the vector and related info
- Architecture:
  - Use Strategy/Factory pattern to support multiple embedding methods (e.g., TF-IDF, Word2Vec, SBERT, OpenAI)
  - Allow easy switching and adding of new embedding implementations

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
│   ├── clustering/         # Clusterer & clustering implementations
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
