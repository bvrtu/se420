"""
Query Interface
Interactive query interface for the RAG system
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from embeddings.embedder import CourseEmbedder
from vector_db.faiss_db import FAISSCourseDB
from rag.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rag_pipeline(data_dir: str = "data", llm_provider: str = "ollama", model_name: str = "llama3.2"):
    """Load RAG pipeline from saved data"""
    data_path = Path(data_dir)
    
    # Initialize components
    embedder = CourseEmbedder(model_name='all-MiniLM-L6-v2')
    embedding_dim = embedder.model.get_sentence_embedding_dimension()
    vector_db = FAISSCourseDB(persist_directory=str(data_path / 'vector_db'), dimension=embedding_dim)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        vector_db=vector_db,
        embedder=embedder,
        llm_provider=llm_provider,
        model_name=model_name
    )
    
    return rag_pipeline


def interactive_query(rag_pipeline):
    """Interactive query interface"""
    print("\n" + "="*60)
    print("RAG-Based Course Intelligence System")
    print("Faculty of Engineering - Izmir University of Economics")
    print("="*60)
    print("\nEnter your questions (type 'exit' to quit, 'help' for examples)")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nExample queries:")
                print("  - What are the core courses in Software Engineering?")
                print("  - Which departments offer machine learning courses?")
                print("  - Compare programming courses in SE and CE")
                print("  - How many elective courses are in the final year?")
                continue
            
            # Run query
            print("\nProcessing query...")
            result = rag_pipeline.query(query, n_results=15)
            
            # Display results
            print("\n" + "-"*60)
            print("ANSWER:")
            print("-"*60)
            print(result['response'])
            
            print("\n" + "-"*60)
            print(f"Retrieved {result['num_results']} relevant chunks")
            print("-"*60)
            
            # Show sources
            if result['retrieved_chunks']:
                print("\nSources:")
                for i, chunk in enumerate(result['retrieved_chunks'][:3], 1):
                    metadata = chunk.get('metadata', {})
                    print(f"  {i}. {metadata.get('course_code', 'Unknown')} - {metadata.get('course_name', 'Unknown')}")
                    print(f"     Department: {metadata.get('department', 'Unknown')}")
                    print(f"     Section: {metadata.get('section', 'Unknown')}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Query RAG system')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--llm-provider', type=str, default='ollama',
                       choices=['ollama', 'huggingface'],
                       help='LLM provider (default: ollama)')
    parser.add_argument('--model-name', type=str, default='llama3.2',
                       help='Model name (default: llama3.2)')
    parser.add_argument('--query', type=str,
                       help='Single query (optional, otherwise interactive mode)')
    
    args = parser.parse_args()
    
    # Load RAG pipeline
    try:
        rag_pipeline = load_rag_pipeline(
            data_dir=args.data_dir,
            llm_provider=args.llm_provider,
            model_name=args.model_name
        )
    except Exception as e:
        logger.error(f"Error loading RAG pipeline: {e}")
        print(f"Error: {str(e)}")
        print("\nMake sure you have run the pipeline first:")
        print("  python main.py")
        sys.exit(1)
    
    # Run query
    if args.query:
        result = rag_pipeline.query(args.query, n_results=15)
        print("\n" + "="*60)
        print("QUERY:", args.query)
        print("="*60)
        print("\nANSWER:")
        print(result['response'])
        print("\n" + "="*60)
    else:
        interactive_query(rag_pipeline)


if __name__ == "__main__":
    main()
