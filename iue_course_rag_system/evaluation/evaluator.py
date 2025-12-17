"""
Evaluation Module
Evaluates RAG system performance using task-driven question sets
"""

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from .metrics import calculate_metrics

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance"""
    
    def __init__(self, rag_pipeline, questions_file: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            rag_pipeline: RAGPipeline instance
            questions_file: Path to JSON file with evaluation questions (optional)
        """
        self.rag_pipeline = rag_pipeline
        self.questions_file = questions_file
        self.questions = []
        
        if questions_file and Path(questions_file).exists():
            self.load_questions(questions_file)
    
    def load_questions(self, questions_file: str):
        """Load evaluation questions from JSON file"""
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.questions = data.get('questions', [])
        logger.info(f"Loaded {len(self.questions)} evaluation questions")
    
    def evaluate_question(self, question: Dict) -> Dict:
        """
        Evaluate a single question
        
        Args:
            question: Question dictionary with 'question', 'category', 'expected_answer', etc.
            
        Returns:
            Evaluation result dictionary
        """
        query = question.get('question', '')
        category = question.get('category', 'unknown')
        expected_answer = question.get('expected_answer', '')
        department_filter = question.get('department_filter')
        course_type_filter = question.get('course_type_filter')
        
        # Run RAG query
        result = self.rag_pipeline.query(
            query=query,
            n_results=question.get('n_results', 5),
            department_filter=department_filter,
            course_type_filter=course_type_filter
        )
        
        # Calculate metrics
        metrics = calculate_metrics(
            expected_answer=expected_answer,
            actual_answer=result.get('response', ''),
            retrieved_chunks=result.get('retrieved_chunks', [])
        )
        
        return {
            'question': query,
            'category': category,
            'expected_answer': expected_answer,
            'actual_answer': result.get('response', ''),
            'retrieved_chunks_count': result.get('num_results', 0),
            'metrics': metrics
        }
    
    def evaluate_all(self, questions: Optional[List[Dict]] = None) -> Dict:
        """
        Evaluate all questions
        
        Args:
            questions: List of questions (optional, uses self.questions if not provided)
            
        Returns:
            Evaluation results dictionary
        """
        if questions is None:
            questions = self.questions
        
        if not questions:
            logger.warning("No questions to evaluate")
            return {}
        
        logger.info(f"Evaluating {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question.get('question', '')[:50]}...")
            result = self.evaluate_question(question)
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'total_questions': len(results),
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all questions"""
        total = len(results)
        if total == 0:
            return {}
        
        # Calculate averages
        avg_retrieval_accuracy = sum(r['metrics'].get('retrieval_accuracy', 0) for r in results) / total
        avg_groundedness = sum(r['metrics'].get('groundedness', 0) for r in results) / total
        
        # Count hallucinations (low groundedness)
        hallucinations = sum(1 for r in results if r['metrics'].get('groundedness', 1) < 0.5)
        hallucination_rate = hallucinations / total
        
        # Count by category
        category_counts = {}
        for result in results:
            category = result.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'average_retrieval_accuracy': avg_retrieval_accuracy,
            'average_groundedness': avg_groundedness,
            'hallucination_rate': hallucination_rate,
            'total_hallucinations': hallucinations,
            'category_distribution': category_counts
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
