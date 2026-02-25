# RAG Pipeline with Streamlit, Ollama, and Qdrant

A complete Retrieval-Augmented Generation (RAG) system with a user-friendly Streamlit interface for document Q&A.

## Features

- Document Upload: Support for PDF, DOCX, and TXT files
- Vector Search: Powered by Qdrant for efficient similarity search
- AI Responses: Uses Ollama for natural language generation
- Interactive Chat: Ask questions and get context-aware answers
- Document Management: Track uploaded files and database statistics

## Prerequisites

1. **Python 3.8+**
2. **uv** - Fast Python package installer
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** - Local LLM runtime
   ```bash
   # Install from https://ollama.ai or use:
   curl -fsSL https://ollama.com/install.sh | sh
   ```

## Installation

Follow these commands one by one:

### 1. Create virtual environment
```bash
uv venv
```

### 2. Activate virtual environment
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv sync
```

### 4. Start Ollama server
Open a new terminal and run:
```bash
ollama serve
```

### 5. Pull your preferred model
In another terminal:
```bash
ollama pull [MODELNAME]
# Or use another model like:
# ollama pull llama2
# ollama pull mistral
```

### 6. Configure the model
Edit `config.py` and set your model:
```python
OLLAMA_MODEL = "[MODELNAME]"  # Change to your pulled model
```

### 7. Run the application
```bash
streamlit run streamlit_ui.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### Run all pipeline phases

On Linux/macOS (bash):
```bash
./pipeline/run_all_phases.sh
```

On Windows (Command Prompt or PowerShell):
```powershell
.\pipeline\run_all_phases.bat
```

### Step 1: Upload Documents
1. Navigate to the "Upload Documents" tab
2. Click "Choose files to upload"
3. Select PDF, DOCX, or TXT files
4. Click "Process and Upload"
5. Wait for processing to complete

### Step 2: Ask Questions
1. Navigate to the "Ask Questions" tab
2. Enter your question in the text input
3. Toggle "Show sources" to view citations
4. Click "Ask" to get an answer
5. View chat history with previous Q&A

### Step 3: Manage Documents
1. Navigate to "Uploaded Files" tab to view all documents
2. Check sidebar for database statistics
3. Use "Clear All Documents" to reset the database

## How It Works

1. **Document Processing**: Files are loaded and split into chunks with overlap
2. **Embedding**: Each chunk is converted to a vector using sentence-transformers
3. **Storage**: Vectors are stored in Qdrant for efficient retrieval
4. **Query**: User questions are embedded and matched against stored vectors
5. **Generation**: Relevant context is sent to Ollama to generate answers

## Troubleshooting

### Ollama connection error
Ensure Ollama is running:
```bash
ollama serve
```

### Model not found
Pull the model first:
```bash
ollama pull [MODELNAME]
```

### Check available models
```bash
ollama list
```

### Memory issues
- Reduce `CHUNK_SIZE` in `config.py`
- Use fewer documents
- Try a smaller model

### Slow responses
- Use a smaller/faster model (e.g., `qwen3:4b` instead of larger models)
- Reduce `TOP_K_RESULTS` in `config.py`

### Port already in use
If Ollama port 11434 is busy, it means Ollama is already running (which is good).

### Qdrant errors
The app uses in-memory Qdrant by default. For persistent storage:
```python
# In config.py
QDRANT_USE_MEMORY = False
```

## License

MIT License
