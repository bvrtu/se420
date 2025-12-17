"""
Embedding Module
Generates semantic embeddings using Sentence-BERT (free, open-source)
"""

import json
from typing import List, Dict
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class CourseEmbedder:
    """Generates embeddings for course content chunks"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedder with Sentence-BERT model
        
        Args:
            model_name: Sentence-BERT model name (default: all-MiniLM-L6-v2 - fast and free)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text:
            # Return zero vector if text is empty
            dim = self.model.get_sentence_embedding_dimension()
            return [0.0] * dim
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Generate embeddings for all chunks
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for embedding generation
            
        Returns:
            List of chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.get('text', '') for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.extend(batch_embeddings.tolist())
            
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            embedded_chunks.append(chunk_copy)
        
        logger.info(f"Embeddings generated for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict], output_path: str):
        """Save embedded chunks to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embeddings saved to {output_path}")
