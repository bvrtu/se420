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

    @staticmethod
    def _trap_passed(actual_answer: str) -> bool:
        """
        Trap questions should be answered negatively (no hallucinated affirmative claims).
        Accept either an explicit "no data / not offered" style answer or a clear "no".
        """
        if not actual_answer:
            return False

        a = actual_answer.lower()

        negative_markers = [
            # Turkish
            "bulunmamaktadır",
            "mevcut değildir",
            "sunmamaktadır",
            "yoktur",
            "hayır",
            # English
            "not available",
            "not offered",
            "does not offer",
            "there is no",
            "no such",
            "no.",
            "no ",
        ]
        affirmative_markers = [
            "evet",
            "vardır",
            "mevcuttur",
            "sunmaktadır",
            "yes",
            "offered",
            "there is a",
        ]

        has_negative = any(m in a for m in negative_markers)
        has_affirmative = any(m in a for m in affirmative_markers)

        # If both appear, treat as failed (ambiguous / potentially hallucinated)
        return has_negative and not has_affirmative
    
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
        
        # Extract course code from query (for hard filtering)
        import re
        pattern = r'\b([A-Z]{2,4})\s*(\d{3,4})\b'
        match = re.search(pattern, query.upper())
        if match:
            code = match.group(1)
            num = match.group(2)
            course_code = f"{code} {num}"
        else:
            course_code = None
        
        # Run RAG query
        result = self.rag_pipeline.query(
            query=query,
            n_results=question.get('n_results', 15),
            department_filter=department_filter,
            course_type_filter=course_type_filter,
            course_code=course_code
        )
        
        actual_answer = result.get('response', '')
        
        # Trap-question check (should be negative)
        is_trap = category == 'trap'
        trap_passed = self._trap_passed(actual_answer) if is_trap else None
        
        # Calculate metrics
        metrics = calculate_metrics(
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            retrieved_chunks=result.get('retrieved_chunks', [])
        )
        
        return {
            'question': query,
            'category': category,
            'expected_answer': expected_answer,
            'actual_answer': actual_answer,
            'retrieved_chunks_count': result.get('num_results', 0),
            'metrics': metrics,
            'is_trap': is_trap
            ,
            'trap_passed': trap_passed
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
            try:
                result = self.evaluate_question(question)
                results.append(result)
            except Exception as e:
                logger.exception(f"Error evaluating question {i}: {e}")
                results.append({
                    'question': question.get('question', ''),
                    'category': question.get('category', 'unknown'),
                    'expected_answer': question.get('expected_answer', ''),
                    'actual_answer': '',
                    'retrieved_chunks_count': 0,
                    'metrics': {},
                    'is_trap': question.get('category') == 'trap',
                    'trap_passed': False if question.get('category') == 'trap' else None,
                    'error': str(e),
                })
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        return {
            'total_questions': len(results),
            'results': results,
            'aggregate_metrics': aggregate_metrics
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all questions."""
        total = len(results)
        if total == 0:
            return {}
        
        # Calculate averages (ignore None values)
        ra_vals = [r.get('metrics', {}).get('retrieval_accuracy') for r in results]
        ra_vals = [v for v in ra_vals if isinstance(v, (int, float))]
        grounded_vals = [r.get('metrics', {}).get('groundedness') for r in results]
        grounded_vals = [v for v in grounded_vals if isinstance(v, (int, float))]
        acc_vals = [r.get('metrics', {}).get('accuracy') for r in results]
        acc_vals = [v for v in acc_vals if isinstance(v, (int, float))]

        avg_retrieval_accuracy = sum(ra_vals) / len(ra_vals) if ra_vals else 0.0
        avg_groundedness = sum(grounded_vals) / len(grounded_vals) if grounded_vals else 0.0
        avg_accuracy = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0

        # Count hallucinations
        # - Trap questions: fail if trap_passed is False
        # - Others: use metrics.is_hallucination
        hallucinations = 0
        for r in results:
            if r.get('category') == 'trap':
                if r.get('trap_passed') is False:
                    hallucinations += 1
            else:
                if r.get('metrics', {}).get('is_hallucination', False):
                    hallucinations += 1
        hallucination_rate = hallucinations / total
        
        # TR+EN format compliance
        tr_en_compliant = sum(1 for r in results if r['metrics'].get('tr_en_format', {}).get('format_compliant', False))
        tr_en_compliance_rate = tr_en_compliant / total if total > 0 else 0.0
        
        # Count by category
        category_counts = {}
        for result in results:
            category = result.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'average_retrieval_accuracy': avg_retrieval_accuracy,
            'average_groundedness': avg_groundedness,
            'average_accuracy': avg_accuracy,
            'retrieval_accuracy_n': len(ra_vals),
            'groundedness_n': len(grounded_vals),
            'accuracy_n': len(acc_vals),
            'hallucination_rate': hallucination_rate,
            'total_hallucinations': hallucinations,
            'tr_en_compliance_rate': tr_en_compliance_rate,
            'total_tr_en_compliant': tr_en_compliant,
            'category_distribution': category_counts
        }
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
