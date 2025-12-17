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


def calculate_accuracy(expected_answer: str, actual_answer: str) -> float:
    """
    Calculate answer accuracy (Görsel: Hallucination / accuracy metrikleri)
    
    Args:
        expected_answer: Expected answer (ground truth)
        actual_answer: Generated answer
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not expected_answer or not actual_answer:
        return 0.0
    
    # Simple keyword-based accuracy
    expected_words = set(re.findall(r'\b\w{3,}\b', expected_answer.lower()))
    actual_words = set(re.findall(r'\b\w{3,}\b', actual_answer.lower()))
    
    if not expected_words:
        return 0.0
    
    overlap = expected_words.intersection(actual_words)
    return len(overlap) / len(expected_words) if expected_words else 0.0


def check_tr_en_format(answer: str) -> Dict[str, bool]:
    """
    Check if answer follows TR+EN format (Görsel: TR+EN enforce testleri)
    
    Args:
        answer: Generated answer
        
    Returns:
        Dictionary with format check results
    """
    answer_lower = answer.lower()
    
    has_tr_section = "answer (tr)" in answer_lower or "cevap (tr)" in answer_lower
    has_en_section = "answer (en)" in answer_lower or "cevap (en)" in answer_lower
    has_separators = "---" in answer or "===" in answer
    
    # Extract TR and EN sections if they exist
    tr_text = ""
    en_text = ""
    
    if has_tr_section and has_en_section:
        # Try to extract TR section
        tr_match = re.search(r'answer\s*\(tr\)[:\-]*\s*(.*?)(?=answer\s*\(en\)|$)', answer, re.IGNORECASE | re.DOTALL)
        if tr_match:
            tr_text = tr_match.group(1).strip()
        
        # Try to extract EN section
        en_match = re.search(r'answer\s*\(en\)[:\-]*\s*(.*?)$', answer, re.IGNORECASE | re.DOTALL)
        if en_match:
            en_text = en_match.group(1).strip()
    
    # Check if both sections have content
    has_tr_content = len(tr_text) > 10
    has_en_content = len(en_text) > 10
    
    return {
        'has_tr_section': has_tr_section,
        'has_en_section': has_en_section,
        'has_separators': has_separators,
        'has_tr_content': has_tr_content,
        'has_en_content': has_en_content,
        'format_compliant': has_tr_section and has_en_section and has_tr_content and has_en_content
    }


def calculate_metrics(expected_answer: str, actual_answer: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Calculate all metrics for a single question (Görsel: Hallucination / accuracy metrikleri eklendi)
    
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
    accuracy = calculate_accuracy(expected_answer, actual_answer)
    
    # TR+EN format check (Görsel: TR+EN enforce testleri)
    tr_en_format = check_tr_en_format(actual_answer)
    
    return {
        'retrieval_accuracy': retrieval_accuracy,
        'groundedness': groundedness,
        'is_hallucination': is_hallucination,
        'accuracy': accuracy,  # Görsel: accuracy metrikleri
        'num_retrieved_chunks': len(retrieved_chunks),
        'tr_en_format': tr_en_format  # Görsel: TR+EN enforce testleri
    }
