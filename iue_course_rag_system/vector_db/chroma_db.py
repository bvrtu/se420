"""
Vector Database Module
Stores and retrieves course embeddings using ChromaDB (free, open-source)
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromaCourseDB:
    """ChromaDB-based vector database for course content"""
    
    def __init__(self, persist_directory: str = "data/vector_db", collection_name: str = "course_chunks"):
        """
        Initialize ChromaDB
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Course content chunks with embeddings"}
        )
        
        logger.info(f"ChromaDB initialized. Collection: {collection_name}")
        logger.info(f"Current collection count: {self.collection.count()}")
    
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
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(embedded_chunks):
            # Generate unique ID
            chunk_id = f"{chunk.get('department', 'unknown')}_{chunk.get('course_code', 'unknown')}_{chunk.get('section', 'unknown')}_{chunk.get('chunk_index', i)}"
            ids.append(chunk_id)
            
            # Get embedding
            embedding = chunk.get('embedding')
            if not embedding:
                logger.warning(f"Chunk {chunk_id} has no embedding, skipping")
                continue
            embeddings.append(embedding)
            
            # Get text
            text = chunk.get('text', '')
            documents.append(text)
            
            # Prepare metadata (ChromaDB only accepts string, int, float, bool)
            metadata = {}
            for key, value in chunk.items():
                if key not in ['embedding', 'text']:
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif value is None:
                        metadata[key] = ""
                    else:
                        metadata[key] = str(value)
            metadatas.append(metadata)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            if (i + batch_size) % (batch_size * 10) == 0:
                logger.info(f"Added {min(i + batch_size, len(ids))}/{len(ids)} chunks")
        
        logger.info(f"Added {len(ids)} chunks to vector database")
        logger.info(f"Total chunks in database: {self.collection.count()}")
    
    def search(self, query_embedding: List[float], n_results: int = 5, 
               department_filter: Optional[str] = None,
               course_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (Mandatory/Elective) (optional)
            
        Returns:
            List of similar chunks with metadata
        """
        # Build where clause for filtering
        where_clause = {}
        if department_filter:
            where_clause['department'] = department_filter
        if course_type_filter:
            where_clause['type'] = course_type_filter
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def reset(self):
        """Reset the collection (delete all data)"""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "Course content chunks with embeddings"}
        )
        logger.info("Vector database reset")
