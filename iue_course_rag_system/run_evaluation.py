"""
Evaluation Script
Runs evaluation on the RAG system using question sets
"""

import argparse
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from embeddings.embedder import CourseEmbedder
from vector_db.faiss_db import FAISSCourseDB
from rag.rag_pipeline import RAGPipeline
from evaluation.evaluator import RAGEvaluator
from evaluation.question_generator import QuestionGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run RAG system evaluation')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--questions-file', type=str,
                       help='Path to questions JSON file (optional, will generate if not provided)')
    parser.add_argument('--output', type=str, default='data/evaluation_results.json',
                       help='Output file for results (default: data/evaluation_results.json)')
    parser.add_argument('--llm-provider', type=str, default='ollama',
                       choices=['ollama', 'huggingface'],
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model-name', type=str, default='llama3.2',
                       help='Model name (default: llama3.2)')
    parser.add_argument('--generate-questions', action='store_true',
                       help='Generate evaluation questions')
    
    args = parser.parse_args()
    
    # Generate questions if requested
    if args.generate_questions or not args.questions_file:
        logger.info("Generating evaluation questions...")
        questions = QuestionGenerator.generate_all_questions()
        questions_file = Path(args.data_dir) / 'evaluation_questions.json'
        QuestionGenerator.save_questions(questions, str(questions_file))
        logger.info(f"Generated {len(questions)} questions and saved to {questions_file}")
        args.questions_file = str(questions_file)
    
    # Load RAG pipeline
    logger.info("Loading RAG pipeline...")
    data_path = Path(args.data_dir)
    
    embedder = CourseEmbedder(model_name='all-MiniLM-L6-v2')
    embedding_dim = embedder.model.get_sentence_embedding_dimension()
    vector_db = FAISSCourseDB(persist_directory=str(data_path / 'vector_db'), dimension=embedding_dim)
    
    rag_pipeline = RAGPipeline(
        vector_db=vector_db,
        embedder=embedder,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        data_dir=args.data_dir
    )
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag_pipeline, questions_file=args.questions_file)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_all()
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {results.get('total_questions', 0)}")
    
    aggregate = results.get('aggregate_metrics', {})
    ra_n = aggregate.get('retrieval_accuracy_n', 0)
    g_n = aggregate.get('groundedness_n', 0)
    acc_n = aggregate.get('accuracy_n', 0)

    if ra_n:
        print(f"\nAverage Retrieval Accuracy: {aggregate.get('average_retrieval_accuracy', 0):.2%} (n={ra_n})")
    else:
        print("\nAverage Retrieval Accuracy: N/A (no ground-truth answers provided)")

    if g_n:
        print(f"Average Groundedness: {aggregate.get('average_groundedness', 0):.2%} (n={g_n})")
    else:
        print("Average Groundedness: N/A")

    if acc_n:
        print(f"Average Accuracy: {aggregate.get('average_accuracy', 0):.2%} (n={acc_n})")
    else:
        print("Average Accuracy: N/A (no ground-truth answers provided)")
    print(f"Hallucination Rate: {aggregate.get('hallucination_rate', 0):.2%}")
    print(f"Total Hallucinations: {aggregate.get('total_hallucinations', 0)}")
    print(f"TR+EN Format Compliance: {aggregate.get('tr_en_compliance_rate', 0):.2%}")
    print(f"Total TR+EN Compliant: {aggregate.get('total_tr_en_compliant', 0)}")
    
    print("\nCategory Distribution:")
    for category, count in aggregate.get('category_distribution', {}).items():
        print(f"  {category}: {count}")
    
    print("\n" + "="*60)
    print(f"Detailed results saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
