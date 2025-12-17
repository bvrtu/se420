"""
Vector Database Module
Stores and retrieves course embeddings using FAISS (free, open-source, reliable)
"""

from .faiss_db import FAISSCourseDB

__all__ = ['FAISSCourseDB']
