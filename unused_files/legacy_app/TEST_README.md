# EV Research Test Runner

This script tests 20 questions from the Sample questions.docx file using:
- **TinyLlama** (local Ollama model) with and without RAG
- **Gemini 2.5** (Google AI model) with and without RAG

The test uses only the `GNEM updated excel (1).xlsx` file for RAG context.

## Setup

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set up Gemini API key:**
   - Copy `.env.example` to `.env`
   - Add your Gemini API key from https://makersuite.google.com/app/apikey
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Ensure TinyLlama is available in Ollama:**
   ```bash
   ollama pull tinyllama
   ```

## Running Tests

```bash
python test_runner.py
```

## Output

The script generates `test_results.xlsx` with columns:
- Question Number
- Question
- TinyLlama RAG (response, time, success status)
- TinyLlama No RAG (response, time, success status)
- Gemini 2.5 RAG (response, time, success status)
- Gemini 2.5 No RAG (response, time, success status)

## How It Works

1. **Extracts 20 questions** from `Sample questions.docx`
2. **Processes the Excel file** into the vector store for RAG
3. **Queries each model configuration** for each question
4. **Collects responses** and timing data
5. **Exports to Excel** for analysis

## Note on RAG Pipeline

The Excel file is structured with columns like:
- Company, Category, Industry Group, Location
- EV Supply Chain Role, Primary OEMs
- Employment, Product/Service, EV/Battery Relevant

Each row is converted to a text chunk with all field data, making it highly retrievable.
