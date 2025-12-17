# Multi-Department RAG-Based Course Intelligence System
## Faculty of Engineering – Izmir University of Economics (IUE)

This project implements an intelligent course intelligence and comparison system for the Faculty of Engineering at IUE using Retrieval-Augmented Generation (RAG) architecture with **100% free, open-source** components.

## Project Structure

```
iue_course_rag_system/
├── scraper/              # Web scraping module
├── data_processing/      # Data cleaning and structuring
├── embeddings/           # Embedding generation (Sentence-BERT - free)
├── vector_db/            # Vector database (ChromaDB - free)
├── rag/                  # RAG pipeline (Ollama/HuggingFace - free)
├── evaluation/           # Evaluation framework
├── data/
│   ├── raw/             # Raw scraped data
│   ├── processed/        # Processed and chunked data
│   ├── embeddings/       # Generated embeddings
│   └── vector_db/        # ChromaDB database
└── main.py               # Main pipeline script
```

## Features

- **Web Scraping**: Automated extraction of course data from IUE ECTS website
- **Data Processing**: Cleaning, normalization, and chunking of course content
- **Semantic Search**: Vector-based semantic search using Sentence-BERT embeddings (free)
- **Vector Database**: FAISS for efficient storage and retrieval (free, reliable)
- **RAG Pipeline**: Question answering using Ollama or HuggingFace (free LLMs)
- **Evaluation**: Comprehensive evaluation framework with metrics

## Target Departments

1. Software Engineering
2. Computer Engineering
3. Electrical and Electronics Engineering
4. Industrial Engineering

## Installation

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama (Recommended for LLM)

Ollama is a free, local LLM runner. Install from: https://ollama.ai

After installation, pull a model:
```bash
ollama pull llama3.2
```

Alternatively, you can use HuggingFace transformers (no additional setup needed, but requires more memory).

## Usage

### 1. Run Complete Pipeline

Scrape, process, embed, and build vector database:

```bash
python main.py
```

Or with options:
```bash
python main.py --departments software_engineering computer_engineering --chunk-size 500
```

### 2. Query the System

Interactive mode:
```bash
python query.py
```

Single query:
```bash
python query.py --query "What are the core courses in Software Engineering?"
```

### 3. Run Evaluation

Generate questions and evaluate:
```bash
python run_evaluation.py --generate-questions
```

Or use existing questions:
```bash
python run_evaluation.py --questions-file data/evaluation_questions.json
```

## Pipeline Steps

1. **Web Scraping**: Extracts course data from IUE ECTS website
2. **Data Processing**: Cleans, normalizes, and chunks course content
3. **Embedding**: Generates semantic embeddings using Sentence-BERT
4. **Vector Database**: Stores embeddings in ChromaDB
5. **RAG Pipeline**: Retrieves relevant chunks and generates answers using free LLM

## Free Components Used

- **Embeddings**: Sentence-BERT (all-MiniLM-L6-v2) - Free, open-source
- **Vector DB**: FAISS - Free, open-source, reliable (no dependency conflicts)
- **LLM**: Ollama (llama3.2) or HuggingFace Transformers - Free, open-source

## Evaluation Categories

The system is evaluated on 5 question categories:
- **A) Single-Department Questions** (10 questions)
- **B) Topic-Based Search** (10 questions)
- **C) Cross-Department Comparison** (10 questions)
- **D) Quantitative/Counting Questions** (10 questions)
- **E) Hallucination/Trap Questions** (20 questions)

## Notes

- All components are free and open-source
- No API keys required
- Ollama runs locally (requires installation)
- FAISS persists data locally (no dependency conflicts)
- Sentence-BERT downloads models automatically on first use
