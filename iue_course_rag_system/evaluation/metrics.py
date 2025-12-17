"""
Metrics Module
Calculates evaluation metrics for RAG system
"""

import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def calculate_retrieval_accuracy(retrieved_chunks: List[Dict], expected_keywords: List[str]) -> float:
    """
    Calculate retrieval accuracy based on keyword matching
    
    Args:
        retrieved_chunks: List of retrieved chunks
        expected_keywords: List of expected keywords in results
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not retrieved_chunks or not expected_keywords:
        return 0.0
    
    # Extract text from all retrieved chunks
    all_text = ' '.join([chunk.get('text', '') for chunk in retrieved_chunks]).lower()
    
    # Count matching keywords
    matches = sum(1 for keyword in expected_keywords if keyword.lower() in all_text)
    
    return matches / len(expected_keywords) if expected_keywords else 0.0


def calculate_groundedness(actual_answer: str, retrieved_chunks: List[Dict]) -> float:
    """
    Calculate groundedness - how well the answer is supported by retrieved chunks
    
    Args:
        actual_answer: Generated answer
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        Groundedness score (0.0 to 1.0)
    """
    if not actual_answer or not retrieved_chunks:
        return 0.0
    
    # Extract key phrases from answer (simple approach)
    answer_words = set(re.findall(r'\b\w{4,}\b', actual_answer.lower()))
    
    if not answer_words:
        return 0.0
    
    # Extract text from all retrieved chunks
    chunk_text = ' '.join([chunk.get('text', '') for chunk in retrieved_chunks]).lower()
    chunk_words = set(re.findall(r'\b\w{4,}\b', chunk_text))
    
    # Calculate overlap
    overlap = answer_words.intersection(chunk_words)
    overlap_ratio = len(overlap) / len(answer_words) if answer_words else 0.0
    
    return min(overlap_ratio, 1.0)


def detect_hallucination(actual_answer: str, retrieved_chunks: List[Dict]) -> bool:
    """
    Detect if answer contains hallucination (unsupported information)
    
    Args:
        actual_answer: Generated answer
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        True if hallucination detected, False otherwise
    """
    if not actual_answer or not retrieved_chunks:
        return True
    
    # Check for common hallucination indicators
    hallucination_phrases = [
        "i don't have information",
        "not available",
        "cannot find",
        "no data",
        "unknown"
    ]
    
    answer_lower = actual_answer.lower()
    if any(phrase in answer_lower for phrase in hallucination_phrases):
        return False  # System correctly says it doesn't know
    
    # Check groundedness
    groundedness = calculate_groundedness(actual_answer, retrieved_chunks)
    
    # Low groundedness indicates potential hallucination
    return groundedness < 0.3


def calculate_metrics(expected_answer: str, actual_answer: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Calculate all metrics for a single question
    
    Args:
        expected_answer: Expected answer (ground truth)
        actual_answer: Generated answer
        retrieved_chunks: List of retrieved chunks
        
    Returns:
        Dictionary with all metrics
    """
    # Extract keywords from expected answer
    expected_keywords = re.findall(r'\b\w{4,}\b', expected_answer.lower()) if expected_answer else []
    
    # Calculate metrics
    retrieval_accuracy = calculate_retrieval_accuracy(retrieved_chunks, expected_keywords)
    groundedness = calculate_groundedness(actual_answer, retrieved_chunks)
    is_hallucination = detect_hallucination(actual_answer, retrieved_chunks)
    
    return {
        'retrieval_accuracy': retrieval_accuracy,
        'groundedness': groundedness,
        'is_hallucination': is_hallucination,
        'num_retrieved_chunks': len(retrieved_chunks)
    }
