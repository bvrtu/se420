from rag.rag_pipeline import RAGPipeline
from vector_db.faiss_db import FAISSCourseDB
from embeddings.embedder import CourseEmbedder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retrieval(rag, query):
    logger.info(f"Query: {query}")
    answer = rag.query(query)
    logger.info(f"Answer: {answer}")
    
    # Also peek at retrieved documents
    logger.info(f"Retrieved {len(retrieved)} chunks")
    for i, doc in enumerate(retrieved):
        # Handle both object and dict access for compatibility
        meta = doc.get('metadata', {}) if isinstance(doc, dict) else doc.metadata
        content = doc.get('page_content', '') if isinstance(doc, dict) else doc.page_content
        
        logger.info(f"Chunk {i+1} Metadata: {meta}")
        logger.info(f"Chunk {i+1} Content Preview: {content[:200]}...")

if __name__ == "__main__":
    # Initialize components
    embedder = CourseEmbedder(model_name='all-MiniLM-L6-v2')
    vector_db = FAISSCourseDB(persist_directory='data/vector_db')
    
    try:
        import ollama
        rag = RAGPipeline(vector_db, embedder, llm_provider="ollama", model_name="llama3.2")
    except:
        rag = RAGPipeline(vector_db, embedder, llm_provider="huggingface", model_name="microsoft/DialoGPT-medium")

    # Test case 1: FR 103 Weekly Topics (SFL)
    logger.info("\n=== Testing FR 103 Weekly Topics ===")
    test_retrieval(rag, "fr 103 weekly topics")

    # Test case 2: GER 101 Weekly Topics (SFL)
    logger.info("\n=== Testing GER 101 Weekly Topics ===")
    test_retrieval(rag, "ger 101 weekly topics")
