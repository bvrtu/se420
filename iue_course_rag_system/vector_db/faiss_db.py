"""
Vector Database Module using FAISS
Stores and retrieves course embeddings using FAISS (free, open-source, reliable)
"""

import faiss
import numpy as np
import json
import pickle
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSCourseDB:
    """FAISS-based vector database for course content"""
    
    def __init__(self, persist_directory: str = "data/vector_db", dimension: int = 384):
        """
        Initialize FAISS database
        
        Args:
            persist_directory: Directory to persist database
            dimension: Dimension of embeddings (default: 384 for all-MiniLM-L6-v2)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        
        # Initialize FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        
        # Store metadata separately
        self.metadata = []
        
        # Load existing data if available
        self.load()
        
        logger.info(f"FAISS database initialized. Dimension: {dimension}")
        logger.info(f"Current index size: {self.index.ntotal}")
    
    def add_chunks(self, embedded_chunks: List[Dict]):
        """
        Add embedded chunks to the database
        
        Args:
            embedded_chunks: List of chunks with 'embedding' field
        """
        if not embedded_chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(embedded_chunks)} chunks to vector database...")
        
        # Prepare embeddings and metadata
        embeddings = []
        new_metadata = []
        
        for chunk in embedded_chunks:
            embedding = chunk.get('embedding')
            if not embedding:
                logger.warning(f"Chunk missing embedding, skipping")
                continue
            
            # Convert to numpy array
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = np.array(embedding, dtype=np.float32)
            
            # Check dimension
            if embedding.shape[0] != self.dimension:
                logger.warning(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.dimension}, skipping")
                continue
            
            embeddings.append(embedding)
            
            # Store metadata (without embedding)
            metadata = {k: v for k, v in chunk.items() if k != 'embedding'}
            new_metadata.append(metadata)
        
        if not embeddings:
            logger.warning("No valid embeddings to add")
            return
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(new_metadata)
        
        logger.info(f"Added {len(embeddings)} chunks to vector database")
        logger.info(f"Total chunks in database: {self.index.ntotal}")
        
        # Save after adding
        self.save()
    
    def search(self, query_embedding: List[float], n_results: int = 5,
               department_filter: Optional[str] = None,
               course_type_filter: Optional[str] = None,
               filters: Optional[Dict] = None,
               boost_section: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (Mandatory/Elective) (optional)
            filters: Dictionary with metadata filters (course_code, department, section) - STRING EQUALITY
            boost_section: Section to boost (e.g., 'credits', 'weekly_topics') - increases relevance score
            
        Returns:
            List of similar chunks with metadata
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Convert query to numpy array
        if isinstance(query_embedding, list):
            query_array = np.array([query_embedding], dtype=np.float32)
        else:
            query_array = np.array([query_embedding], dtype=np.float32)
        
        # Check dimension
        if query_array.shape[1] != self.dimension:
            logger.error(f"Query dimension mismatch: {query_array.shape[1]} != {self.dimension}")
            return []
        
        # Search (get more results for better coverage)
        # If filtering, get more results to ensure we have enough after filtering
        k = min(n_results * 5, self.index.ntotal) if (department_filter or course_type_filter or filters) else n_results * 2
        k = min(k, self.index.ntotal)
        k = max(k, n_results)  # At least n_results
        
        distances, indices = self.index.search(query_array, k)
        
        # Format results with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            if idx >= len(self.metadata):
                logger.warning(f"Index {idx} out of range for metadata")
                continue
            
            metadata = self.metadata[idx]
            
            # Apply filters (STRING EQUALITY - semantic değil)
            if filters:
                # course_code exact match
                if 'course_code' in filters:
                    filter_code = filters['course_code'].replace(' ', '').upper()
                    chunk_code = metadata.get('course_code', '').replace(' ', '').upper()
                    if filter_code != chunk_code:
                        continue
                
                # department exact match
                if 'department' in filters:
                    if metadata.get('department', '') != filters['department']:
                        continue
                
                # section exact match
                if 'section' in filters:
                    if metadata.get('section', '').lower() != filters['section'].lower():
                        continue
            
            # Apply legacy filters
            if department_filter and metadata.get('department') != department_filter:
                continue
            if course_type_filter and metadata.get('type') != course_type_filter:
                continue
            
            # Get text from metadata
            text = metadata.get('text', '')
            
            # Calculate similarity score (lower distance = higher similarity)
            similarity_score = 1.0 / (1.0 + float(distances[0][i]))
            
            # Section boost (ÇÖZÜM B: Section-aware retrieval)
            if boost_section:
                chunk_section = metadata.get('section', '').lower()
                if chunk_section == boost_section.lower():
                    similarity_score *= 1.5  # Boost matching sections
            
            result = {
                'id': f"chunk_{idx}",
                'text': text,
                'metadata': metadata,
                'distance': float(distances[0][i]),
                'similarity': similarity_score
            }
            results.append(result)
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top n_results
        return results[:n_results]
    
    def save(self):
        """Save index and metadata to disk"""
        # Save FAISS index
        index_path = self.persist_directory / 'faiss.index'
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = self.persist_directory / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.debug(f"Saved FAISS index and metadata to {self.persist_directory}")
    
    def load(self):
        """Load index and metadata from disk"""
        index_path = self.persist_directory / 'faiss.index'
        metadata_path = self.persist_directory / 'metadata.pkl'
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                logger.info(f"Loaded {len(self.metadata)} metadata entries")
            except Exception as e:
                logger.error(f"Error loading FAISS database: {e}")
                # Reinitialize on error
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = []
        else:
            logger.info("No existing FAISS database found, starting fresh")
    
    def reset(self):
        """Reset the database (delete all data)"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        
        # Delete saved files
        index_path = self.persist_directory / 'faiss.index'
        metadata_path = self.persist_directory / 'metadata.pkl'
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        
        logger.info("FAISS database reset")
