# Setup Guide

## Quick Start

### 1. Create Virtual Environment

```bash
cd iue_course_rag_system
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This may take a few minutes as it downloads:
- Sentence-BERT models (automatic on first use)
- PyTorch (for embeddings)

### 3. Install Ollama (Recommended for LLM)

Ollama is a free, local LLM runner. Install from: https://ollama.ai

After installation, pull a model:
```bash
ollama pull llama3.2
```

**Alternative:** If you don't want to install Ollama, the system will use HuggingFace Transformers (requires more memory).

### 4. Run the Pipeline

```bash
# Scrape, process, embed, and build vector database
python main.py
```

This will:
1. Scrape course data from all 4 departments
2. Process and chunk the data
3. Generate embeddings
4. Build vector database
5. Initialize RAG pipeline

**Time estimate:** 
- Scraping: ~10-15 minutes (with 1s delay)
- Processing: ~1 minute
- Embedding: ~5-10 minutes (depending on CPU/GPU)
- Vector DB: ~1 minute

### 5. Query the System

```bash
# Interactive mode
python query.py

# Single query
python query.py --query "What are the core courses in Software Engineering?"
```

### 6. Run Evaluation

```bash
# Generate questions and evaluate
python run_evaluation.py --generate-questions
```

## Project Structure

```
iue_course_rag_system/
├── scraper/              # Web scraping (complete, tested)
├── data_processing/      # Data cleaning and chunking
├── embeddings/           # Sentence-BERT embeddings (free)
├── vector_db/            # FAISS vector database (free)
├── rag/                  # RAG pipeline (Ollama/HuggingFace - free)
├── evaluation/           # Evaluation framework
├── data/                 # Generated data (created after running)
│   ├── raw/             # Scraped JSON
│   ├── processed/       # Chunked JSON
│   ├── embeddings/       # Embedded chunks (optional; can be regenerated)
│   └── vector_db/        # FAISS index + metadata
├── main.py              # Main pipeline script
├── query.py             # Query interface
└── run_evaluation.py    # Evaluation script
```

## Free Components

All components are **100% free and open-source**:

- ✅ **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2)
- ✅ **Vector DB**: FAISS
- ✅ **LLM**: Ollama (llama3.2) or HuggingFace Transformers
- ✅ **No API keys required**
- ✅ **No paid services**

## Troubleshooting

### FAISS Installation

FAISS is used as the vector database (reliable and easy to install).
If you encounter issues:
- For CPU: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu` (requires CUDA)

### Ollama Not Found

If Ollama is not installed, the system will automatically use HuggingFace Transformers. This requires more memory but works without additional setup.

### Memory Issues

If you run out of memory:
- Use a smaller embedding model: `--embedding-model all-MiniLM-L6-v2` (default)
- Reduce chunk size: `--chunk-size 300`
- Use CPU instead of GPU for embeddings

## Next Steps

1. Run the pipeline: `python main.py`
2. Test queries: `python query.py`
3. Run evaluation: `python run_evaluation.py --generate-questions`
