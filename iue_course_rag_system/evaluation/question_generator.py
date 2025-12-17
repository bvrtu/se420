"""
Question Generator Module
Generates evaluation questions based on report categories
"""

import json
from typing import List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates evaluation questions for different categories"""
    
    @staticmethod
    def generate_single_department_questions() -> List[Dict]:
        """Generate single-department questions (Category A)"""
        return [
            {
                "category": "single_department",
                "question": "What are the core courses in the second year of Software Engineering?",
                "department_filter": "Software Engineering",
                "expected_answer": "Core courses include Data Structures, Discrete Mathematics, Software Engineering concepts",
                "n_results": 5
            },
            {
                "category": "single_department",
                "question": "Which mandatory courses are offered in the fifth year of Computer Engineering?",
                "department_filter": "Computer Engineering",
                "course_type_filter": "Mandatory",
                "expected_answer": "Fifth year courses include capstone projects and final year courses",
                "n_results": 5
            },
            {
                "category": "single_department",
                "question": "Which courses focus on optimization techniques in Industrial Engineering?",
                "department_filter": "Industrial Engineering",
                "expected_answer": "Courses related to optimization, operations research, and mathematical modeling",
                "n_results": 5
            }
        ]
    
    @staticmethod
    def generate_topic_based_questions() -> List[Dict]:
        """Generate topic-based search questions (Category B)"""
        return [
            {
                "category": "topic_based",
                "question": "Which engineering departments offer machine-learning-related courses?",
                "expected_answer": "Multiple departments offer ML courses including Computer Engineering and Software Engineering",
                "n_results": 10
            },
            {
                "category": "topic_based",
                "question": "Which courses cover signal processing topics?",
                "expected_answer": "Signal processing courses are typically in Electrical and Electronics Engineering",
                "n_results": 5
            }
        ]
    
    @staticmethod
    def generate_comparison_questions() -> List[Dict]:
        """Generate cross-department comparison questions (Category C)"""
        return [
            {
                "category": "cross_department_comparison",
                "question": "List programming-focused courses in Software Engineering and Computer Engineering",
                "departments": ["Software Engineering", "Computer Engineering"],
                "expected_answer": "Both departments offer programming courses with different focuses",
                "n_results": 10
            },
            {
                "category": "cross_department_comparison",
                "question": "Which department emphasizes data analysis more strongly?",
                "departments": ["all"],
                "expected_answer": "Comparison of data analysis courses across departments",
                "n_results": 10
            }
        ]
    
    @staticmethod
    def generate_quantitative_questions() -> List[Dict]:
        """Generate quantitative/counting questions (Category D)"""
        return [
            {
                "category": "quantitative",
                "question": "How many elective courses are offered in the final year of Software Engineering?",
                "department_filter": "Software Engineering",
                "course_type_filter": "Elective",
                "expected_answer": "Number of elective courses in final year",
                "n_results": 10
            },
            {
                "category": "quantitative",
                "question": "Which academic year has the highest total ECTS load?",
                "expected_answer": "Academic year with highest ECTS credits",
                "n_results": 20
            }
        ]
    
    @staticmethod
    def generate_trap_questions() -> List[Dict]:
        """Generate hallucination/trap questions (Category E)"""
        return [
            {
                "category": "trap",
                "question": "Does the Software Engineering department offer a course on Quantum Thermodynamics?",
                "department_filter": "Software Engineering",
                "expected_answer": "No",
                "n_results": 5
            },
            {
                "category": "trap",
                "question": "Is there a compulsory course on Astrophysics in Computer Engineering?",
                "department_filter": "Computer Engineering",
                "course_type_filter": "Mandatory",
                "expected_answer": "No",
                "n_results": 5
            }
        ]
    
    @staticmethod
    def generate_all_questions() -> List[Dict]:
        """Generate all evaluation questions"""
        all_questions = []
        all_questions.extend(QuestionGenerator.generate_single_department_questions())
        all_questions.extend(QuestionGenerator.generate_topic_based_questions())
        all_questions.extend(QuestionGenerator.generate_comparison_questions())
        all_questions.extend(QuestionGenerator.generate_quantitative_questions())
        all_questions.extend(QuestionGenerator.generate_trap_questions())
        return all_questions
    
    @staticmethod
    def save_questions(questions: List[Dict], output_path: str):
        """Save questions to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "questions": questions,
            "total_questions": len(questions),
            "categories": {
                "single_department": len([q for q in questions if q.get('category') == 'single_department']),
                "topic_based": len([q for q in questions if q.get('category') == 'topic_based']),
                "cross_department_comparison": len([q for q in questions if q.get('category') == 'cross_department_comparison']),
                "quantitative": len([q for q in questions if q.get('category') == 'quantitative']),
                "trap": len([q for q in questions if q.get('category') == 'trap'])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(questions)} questions to {output_path}")
