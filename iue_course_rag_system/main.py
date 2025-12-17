"""
Main Pipeline Script
Complete RAG-based course intelligence system pipeline
1. Scrape course data
2. Process and chunk data
3. Create embeddings
4. Build vector database
5. Initialize RAG pipeline
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scraper.scraper import IUECourseScraper
from data_processing.processor import CourseDataProcessor
from embeddings.embedder import CourseEmbedder
from vector_db.faiss_db import FAISSCourseDB
from rag.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(
    departments: List[str] = ['all'],
    output_dir: str = 'data',
    embedding_model: str = 'all-MiniLM-L6-v2',
    delay: float = 1.0,
    chunk_size: int = 500,
    skip_scraping: bool = False,
    skip_processing: bool = False,
    skip_embedding: bool = False,
    skip_vector_db: bool = False
):
    """
    Run the complete pipeline
    
    Args:
        departments: List of departments to process (or ['all'])
        output_dir: Output directory for all data
        embedding_model: Sentence transformers model name
        delay: Delay between scraping requests
        chunk_size: Size of text chunks
        skip_scraping: Skip scraping if data already exists
        skip_processing: Skip processing if data already exists
        skip_embedding: Skip embedding if data already exists
        skip_vector_db: Skip vector DB if already exists
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scrape course data
    raw_data_path = output_path / 'raw' / 'scraped_courses.json'
    
    if not skip_scraping or not raw_data_path.exists():
        logger.info("="*60)
        logger.info("STEP 1: Scraping Course Data")
        logger.info("="*60)
        
        scraper = IUECourseScraper(delay=delay)
        
        if 'all' in departments:
            logger.info("Scraping all departments...")
            all_data = scraper.scrape_all_departments()
        else:
            all_data = {}
            for dept in departments:
                logger.info(f"Scraping department: {dept}")
                courses = scraper.scrape_department(dept)
                all_data[dept] = courses
        
        # Save raw data
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        scraper.save_to_json(all_data, str(raw_data_path))
        logger.info(f"Raw data saved to {raw_data_path}")
    else:
        logger.info(f"Loading existing raw data from {raw_data_path}")
        with open(raw_data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    
    # Step 2: Process and chunk data
    logger.info("="*60)
    logger.info("STEP 2: Processing and Chunking Data")
    logger.info("="*60)
    
    processed_data_path = output_path / 'processed' / 'chunks.json'
    
    if not skip_processing or not processed_data_path.exists():
        processor = CourseDataProcessor(chunk_size=chunk_size)
        chunks = processor.process_all_courses(all_data)
        
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        processor.save_processed_data(chunks, str(processed_data_path))
        logger.info(f"Processed {len(chunks)} chunks saved to {processed_data_path}")
    else:
        logger.info(f"Loading existing processed data from {processed_data_path}")
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks")
    
    # Step 3: Create embeddings
    logger.info("="*60)
    logger.info("STEP 3: Creating Embeddings")
    logger.info("="*60)
    
    embedder = CourseEmbedder(model_name=embedding_model)
    
    embedded_data_path = output_path / 'embeddings' / 'embedded_chunks.json'
    
    if not skip_embedding or not embedded_data_path.exists():
        embedded_chunks = embedder.embed_chunks(chunks, batch_size=32)
        
        embedded_data_path.parent.mkdir(parents=True, exist_ok=True)
        embedder.save_embeddings(embedded_chunks, str(embedded_data_path))
        logger.info(f"Embedded chunks saved to {embedded_data_path}")
    else:
        logger.info(f"Loading existing embeddings from {embedded_data_path}")
        with open(embedded_data_path, 'r', encoding='utf-8') as f:
            embedded_chunks = json.load(f)
        logger.info(f"Loaded {len(embedded_chunks)} embedded chunks")
    
    # Step 4: Build vector database
    logger.info("="*60)
    logger.info("STEP 4: Building Vector Database")
    logger.info("="*60)
    
    vector_db_path = output_path / 'vector_db'
    
    # Get embedding dimension from embedder
    embedding_dim = embedder.model.get_sentence_embedding_dimension()
    
    if not skip_vector_db:
        vector_db = FAISSCourseDB(persist_directory=str(vector_db_path), dimension=embedding_dim)
        
        # Check if database is empty
        if vector_db.index.ntotal == 0:
            logger.info("Vector database is empty, adding chunks...")
            vector_db.add_chunks(embedded_chunks)
        else:
            logger.info(f"Vector database already contains {vector_db.index.ntotal} chunks")
    else:
        vector_db = FAISSCourseDB(persist_directory=str(vector_db_path), dimension=embedding_dim)
        logger.info(f"Using existing vector database with {vector_db.index.ntotal} chunks")
    
    # Step 5: Initialize RAG Pipeline
    logger.info("="*60)
    logger.info("STEP 5: Initializing RAG Pipeline")
    logger.info("="*60)
    
    # Use Ollama if available, otherwise HuggingFace
    try:
        import ollama
        llm_provider = "ollama"
        model_name = "llama3.2"
        logger.info("Using Ollama for LLM (free, local)")
    except ImportError:
        llm_provider = "huggingface"
        model_name = "microsoft/DialoGPT-medium"
        logger.info("Using HuggingFace Transformers for LLM (free)")
    
    rag_pipeline = RAGPipeline(
        vector_db=vector_db,
        embedder=embedder,
        llm_provider=llm_provider,
        model_name=model_name,
        data_dir=str(output_path)  # Görsel: Dataset fallback için data_dir ekle
    )
    
    logger.info("="*60)
    logger.info("Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Vector database chunks: {vector_db.index.ntotal}")
    logger.info(f"RAG Pipeline ready for queries")
    
    return rag_pipeline, vector_db, embedder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run complete RAG pipeline')
    parser.add_argument('--departments', nargs='+', default=['all'],
                       help='Departments to process (default: all)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--embedding-model', type=str, default='all-MiniLM-L6-v2',
                       help='Sentence-BERT model name (default: all-MiniLM-L6-v2)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between scraping requests (default: 1.0)')
    parser.add_argument('--chunk-size', type=int, default=500,
                       help='Chunk size in characters (default: 500)')
    parser.add_argument('--skip-scraping', action='store_true',
                       help='Skip scraping step')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip processing step')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='Skip embedding step')
    parser.add_argument('--skip-vector-db', action='store_true',
                       help='Skip vector DB step')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        departments=args.departments,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        delay=args.delay,
        chunk_size=args.chunk_size,
        skip_scraping=args.skip_scraping,
        skip_processing=args.skip_processing,
        skip_embedding=args.skip_embedding,
        skip_vector_db=args.skip_vector_db
    )
