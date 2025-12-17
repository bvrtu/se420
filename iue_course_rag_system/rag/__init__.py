"""
RAG Pipeline Module
Retrieval-Augmented Generation for course intelligence and comparison
Uses free, open-source LLM (Ollama or HuggingFace)
"""

from .rag_pipeline import RAGPipeline
from .answer_formatter import format_answer, build_no_data_answer

__all__ = ['RAGPipeline', 'format_answer', 'build_no_data_answer']
