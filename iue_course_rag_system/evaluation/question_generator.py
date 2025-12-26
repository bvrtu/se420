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
    def _q(
        category: str,
        question: str,
        expected_answer: str = "",
        n_results: int = 10,
        department_filter: str | None = None,
        course_type_filter: str | None = None,
    ) -> Dict:
        """Build a standardized question dict."""
        q: Dict = {
            "category": category,
            "question": question,
            "expected_answer": expected_answer or "Manually verify against official curriculum data.",
            "n_results": n_results,
        }
        if department_filter:
            q["department_filter"] = department_filter
        if course_type_filter:
            q["course_type_filter"] = course_type_filter
        return q
    
    @staticmethod
    def generate_single_department_questions() -> List[Dict]:
        """Generate single-department questions (Category A)"""
        # Required count in the report: 10
        return [
            # Core / year-based
            QuestionGenerator._q(
                "single_department",
                "What are the core courses in the second year of Software Engineering?",
                n_results=12,
                department_filter="Software Engineering",
            ),
            QuestionGenerator._q(
                "single_department",
                "What are the core courses in the third year of Computer Engineering?",
                n_results=12,
                department_filter="Computer Engineering",
            ),
            QuestionGenerator._q(
                "single_department",
                "What are the core courses in the second year of Electrical and Electronics Engineering?",
                n_results=12,
                department_filter="Electrical and Electronics Engineering",
            ),
            QuestionGenerator._q(
                "single_department",
                "What are the core courses in the third year of Industrial Engineering?",
                n_results=12,
                department_filter="Industrial Engineering",
            ),
            # Mandatory lists by year
            QuestionGenerator._q(
                "single_department",
                "Which mandatory courses are offered in the fifth year of Computer Engineering?",
                n_results=12,
                department_filter="Computer Engineering",
                course_type_filter="Mandatory",
            ),
            QuestionGenerator._q(
                "single_department",
                "Which mandatory courses are offered in the fourth year of Software Engineering?",
                n_results=12,
                department_filter="Software Engineering",
                course_type_filter="Mandatory",
            ),
            QuestionGenerator._q(
                "single_department",
                "Which mandatory courses are offered in the fourth year of Industrial Engineering?",
                n_results=12,
                department_filter="Industrial Engineering",
                course_type_filter="Mandatory",
            ),
            # Topic focus inside one department
            QuestionGenerator._q(
                "single_department",
                "Which courses focus on optimization techniques in Industrial Engineering?",
                n_results=12,
                department_filter="Industrial Engineering",
            ),
            QuestionGenerator._q(
                "single_department",
                "Which courses focus on embedded systems in Electrical and Electronics Engineering?",
                n_results=12,
                department_filter="Electrical and Electronics Engineering",
            ),
            QuestionGenerator._q(
                "single_department",
                "Which courses focus on software testing in Software Engineering?",
                n_results=12,
                department_filter="Software Engineering",
            ),
        ]
    
    @staticmethod
    def generate_topic_based_questions() -> List[Dict]:
        """Generate topic-based search questions (Category B)"""
        # Required count in the report: 10
        return [
            QuestionGenerator._q("topic_based", "Which engineering departments offer machine-learning-related courses?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses cover signal processing topics?", n_results=15),
            QuestionGenerator._q("topic_based", "Which courses are related to data analysis or data mining?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses mention optimization or operations research in their description/objectives?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses are about computer networks or networking?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses focus on cybersecurity or information security?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses are related to software engineering project management?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses cover artificial intelligence topics?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses are about databases or data management?", n_results=20),
            QuestionGenerator._q("topic_based", "Which courses focus on embedded systems or microcontrollers?", n_results=20),
        ]
    
    @staticmethod
    def generate_comparison_questions() -> List[Dict]:
        """Generate cross-department comparison questions (Category C)"""
        # Required count in the report: 10
        return [
            QuestionGenerator._q("cross_department_comparison", "List programming-focused courses in Software Engineering and Computer Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Which department emphasizes data analysis more strongly?", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare machine learning related course offerings between Software Engineering and Computer Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare signal processing course coverage between Electrical and Electronics Engineering and Computer Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare optimization-related courses between Industrial Engineering and Software Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Which department offers more courses related to embedded systems: Electrical and Electronics Engineering or Computer Engineering?", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare database-related courses between Software Engineering and Computer Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare cybersecurity-related course offerings between Software Engineering and Computer Engineering", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Which department appears to have a stronger focus on project-based courses: Software Engineering or Industrial Engineering?", n_results=20),
            QuestionGenerator._q("cross_department_comparison", "Compare capstone/graduation project style courses across all engineering departments", n_results=25),
        ]
    
    @staticmethod
    def generate_quantitative_questions() -> List[Dict]:
        """Generate quantitative/counting questions (Category D)"""
        # Required count in the report: 10
        return [
            QuestionGenerator._q(
                "quantitative",
                "How many elective courses are offered in the final year of Software Engineering?",
                n_results=20,
                department_filter="Software Engineering",
                course_type_filter="Elective",
            ),
            QuestionGenerator._q(
                "quantitative",
                "How many elective courses are offered in the final year of Computer Engineering?",
                n_results=20,
                department_filter="Computer Engineering",
                course_type_filter="Elective",
            ),
            QuestionGenerator._q(
                "quantitative",
                "How many elective courses are offered in the final year of Electrical and Electronics Engineering?",
                n_results=20,
                department_filter="Electrical and Electronics Engineering",
                course_type_filter="Elective",
            ),
            QuestionGenerator._q(
                "quantitative",
                "How many elective courses are offered in the final year of Industrial Engineering?",
                n_results=20,
                department_filter="Industrial Engineering",
                course_type_filter="Elective",
            ),
            QuestionGenerator._q("quantitative", "Which academic year has the highest total ECTS load?", n_results=25),
            QuestionGenerator._q("quantitative", "How many mandatory courses are offered in the second year of Software Engineering?", n_results=20, department_filter="Software Engineering", course_type_filter="Mandatory"),
            QuestionGenerator._q("quantitative", "How many mandatory courses are offered in the third year of Computer Engineering?", n_results=20, department_filter="Computer Engineering", course_type_filter="Mandatory"),
            QuestionGenerator._q("quantitative", "How many courses (mandatory + elective) exist in the dataset for Software Engineering?", n_results=25, department_filter="Software Engineering"),
            QuestionGenerator._q("quantitative", "How many courses in the dataset are labeled as Elective across all departments?", n_results=25),
            QuestionGenerator._q("quantitative", "How many distinct course codes are present in the dataset across all departments?", n_results=25),
        ]
    
    @staticmethod
    def generate_trap_questions() -> List[Dict]:
        """Generate hallucination/trap questions (Category E)"""
        # Required count in the report: 20 (all should be answered negatively)
        traps = [
            ("Software Engineering", None, "Does the Software Engineering department offer a course on Quantum Thermodynamics?"),
            ("Computer Engineering", "Mandatory", "Is there a compulsory course on Astrophysics in Computer Engineering?"),
            ("Electrical and Electronics Engineering", None, "Does Electrical and Electronics Engineering offer a course on Ancient Greek Philosophy?"),
            ("Industrial Engineering", None, "Is there a course on Medieval Literature in Industrial Engineering?"),
            ("Software Engineering", None, "Does Software Engineering offer a course titled 'Dragon Engineering'?"),
            ("Computer Engineering", None, "Is there a course on Marine Biology in Computer Engineering?"),
            ("Electrical and Electronics Engineering", "Mandatory", "Is there a mandatory course on Zoology in Electrical and Electronics Engineering?"),
            ("Industrial Engineering", "Mandatory", "Is there a mandatory course on Archaeology in Industrial Engineering?"),
            ("Software Engineering", None, "Does Software Engineering offer a course on Quantum Field Theory?"),
            ("Computer Engineering", None, "Does Computer Engineering offer a course on Black Hole Thermodynamics?"),
            ("Electrical and Electronics Engineering", None, "Is there a course on Renaissance Art History in Electrical and Electronics Engineering?"),
            ("Industrial Engineering", None, "Is there a course on Astrobiology in Industrial Engineering?"),
            ("Software Engineering", "Mandatory", "Is there a mandatory course on Meteorology in Software Engineering?"),
            ("Computer Engineering", "Mandatory", "Is there a compulsory course on Botany in Computer Engineering?"),
            ("Electrical and Electronics Engineering", None, "Does Electrical and Electronics Engineering offer a course on Mythology Studies?"),
            ("Industrial Engineering", None, "Is there a course on Culinary Arts in Industrial Engineering?"),
            ("Software Engineering", None, "Does Software Engineering offer a course on Veterinary Medicine?"),
            ("Computer Engineering", None, "Is there a course on Oceanography in Computer Engineering?"),
            ("Electrical and Electronics Engineering", None, "Is there a course on Paleontology in Electrical and Electronics Engineering?"),
            ("Industrial Engineering", None, "Does Industrial Engineering offer a course on Space Archaeology?"),
        ]

        out: List[Dict] = []
        for dept, ctype, qtext in traps:
            out.append(
                QuestionGenerator._q(
                    "trap",
                    qtext,
                    expected_answer="No",
                    n_results=10,
                    department_filter=dept,
                    course_type_filter=ctype,
                )
            )
        return out
    
    @staticmethod
    def generate_all_questions() -> List[Dict]:
        """Generate all evaluation questions"""
        # Target counts (as in the report): 10/10/10/10/20 = 60 total
        a = QuestionGenerator.generate_single_department_questions()
        b = QuestionGenerator.generate_topic_based_questions()
        c = QuestionGenerator.generate_comparison_questions()
        d = QuestionGenerator.generate_quantitative_questions()
        e = QuestionGenerator.generate_trap_questions()

        assert len(a) == 10, f"Category A must be 10 questions, got {len(a)}"
        assert len(b) == 10, f"Category B must be 10 questions, got {len(b)}"
        assert len(c) == 10, f"Category C must be 10 questions, got {len(c)}"
        assert len(d) == 10, f"Category D must be 10 questions, got {len(d)}"
        assert len(e) == 20, f"Category E must be 20 questions, got {len(e)}"

        return [*a, *b, *c, *d, *e]
    
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
