"""
Embedding Module
Generates semantic embeddings using Sentence-BERT (free, open-source)
"""

import json
import re
from typing import List, Dict
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Maximum chunk size for embedding quality (ÇÖZÜM: Çok uzun chunk → embedding kalitesi düşüyor)
MAX_CHUNK_SIZE = 500


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
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of maximum size (ÇÖZÜM: Retrieval precision artırır)
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= MAX_CHUNK_SIZE:
            return [text]
        
        chunks = []
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > MAX_CHUNK_SIZE and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
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
        
        # Split if too long (ÇÖZÜM: Embedding kalitesi için)
        if len(text) > MAX_CHUNK_SIZE:
            # Use first chunk for embedding
            chunks = self.split_text(text)
            text = chunks[0] if chunks else text
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Generate embeddings for all chunks (Görsel: Chunk uzunluğu kontrolü çok önemli)
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for embedding generation
            
        Returns:
            List of chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Görsel: Chunk uzunluğu kontrolü (700-1000 token → embedding bulanıklaşır)
        # Görsel: MAX_CHUNK_SIZE = 500 gibi bir sınır öneririm
        processed_chunks = []
        for chunk in chunks:
            text = chunk.get('text', '')
            # Görsel: Uzun chunk'ları split et (embedding quality için)
            if len(text) > MAX_CHUNK_SIZE:
                logger.warning(f"Chunk too long ({len(text)} chars), splitting...")
                split_texts = self.split_text(text)
                # Her split için ayrı chunk oluştur
                for i, split_text in enumerate(split_texts):
                    chunk_copy = chunk.copy()
                    chunk_copy['text'] = split_text
                    chunk_copy['chunk_index'] = chunk.get('chunk_index', 0) * 1000 + i  # Sub-index
                    processed_chunks.append(chunk_copy)
            else:
                processed_chunks.append(chunk)
        
        # Extract texts
        texts = [chunk.get('text', '') for chunk in processed_chunks]
        
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
        for chunk, embedding in zip(processed_chunks, all_embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            embedded_chunks.append(chunk_copy)
        
        logger.info(f"Embeddings generated for {len(embedded_chunks)} chunks (original: {len(chunks)})")
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict], output_path: str):
        """Save embedded chunks to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Embeddings saved to {output_path}")
