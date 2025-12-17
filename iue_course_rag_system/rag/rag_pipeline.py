"""
RAG Pipeline Module
Retrieval-Augmented Generation for course intelligence and comparison
Uses free, open-source LLM (Ollama or HuggingFace Transformers)
"""

import json
import logging
import re
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Ollama (free, local LLM)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Install with: pip install ollama")

# Try to import HuggingFace transformers as fallback
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers torch")


class RAGPipeline:
    """RAG Pipeline for course intelligence and comparison"""
    
    def __init__(self, vector_db, embedder, llm_provider: str = "ollama", model_name: str = "llama3.2"):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_db: FAISSCourseDB instance
            embedder: CourseEmbedder instance
            llm_provider: LLM provider ("ollama" or "huggingface")
            model_name: Model name (e.g., "llama3.2" for Ollama, "microsoft/DialoGPT-medium" for HuggingFace)
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        # Initialize LLM
        if llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama not available. Install with: pip install ollama")
            self.llm = None  # Ollama uses API calls
            logger.info(f"Using Ollama with model: {model_name}")
        elif llm_provider == "huggingface":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers not available. Install with: pip install transformers torch")
            logger.info(f"Loading HuggingFace model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("HuggingFace model loaded")
        else:
            raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    def retrieve(self, query: str, n_results: int = 15,
                 department_filter: Optional[str] = None,
                 course_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (optional)
            
        Returns:
            List of relevant chunks
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector database
        results = self.vector_db.search(
            query_embedding=query_embedding,
            n_results=n_results,
            department_filter=department_filter,
            course_type_filter=course_type_filter
        )
        
        return results
    
    def generate_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Generate context string from retrieved chunks
        
        Args:
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant course information found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '')
            
            context_parts.append(f"[Source {i}]")
            
            # Always include course code and name first (most important for matching)
            course_code = metadata.get('course_code', 'Unknown')
            course_name = metadata.get('course_name', 'Unknown')
            context_parts.append(f"Course Code: {course_code}")
            context_parts.append(f"Course Name: {course_name}")
            context_parts.append(f"Department: {metadata.get('department', 'Unknown')}")
            
            # Add ALL important metadata (always show, even if None/empty for clarity)
            course_type = metadata.get('type', '')
            context_parts.append(f"Type: {course_type if course_type else 'Not specified'}")  # Mandatory or Elective
            
            ects = metadata.get('ects')
            # Show ECTS even if None - important for LLM to know it was checked
            if ects is not None:
                context_parts.append(f"ECTS Credits: {ects}")
            else:
                context_parts.append(f"ECTS Credits: Not specified")
            
            local_credits = metadata.get('local_credits')
            if local_credits is not None:
                context_parts.append(f"Local Credits: {local_credits}")
            else:
                context_parts.append(f"Local Credits: Not specified")
            
            semester = metadata.get('semester', '')
            if semester:
                context_parts.append(f"Semester: {semester}")
            
            year = metadata.get('year')
            if year:
                context_parts.append(f"Year: {year}")
            
            section = metadata.get('section', '')
            if section:
                context_parts.append(f"Section: {section}")
            
            # Add content (may contain additional course information)
            context_parts.append(f"Content: {text}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated response
        """
        # Build prompt
        prompt = f"""You are an intelligent course information assistant for the Faculty of Engineering at Izmir University of Economics.

Based on the following course information, answer the user's question accurately and comprehensively.

Course Information:
{context}

User Question: {query}

CRITICAL INSTRUCTIONS:
- Answer based ONLY on the provided course information
- Pay attention to ALL metadata fields in each source:
  * Course Code (e.g., "SE 115", "FR 103", "CE 304", "SFL 1013")
  * Course Name
  * Department (Software Engineering, Computer Engineering, Electrical and Electronics Engineering, Industrial Engineering)
  * Type: "Mandatory" or "Elective" (this is CRITICAL for filtering)
  * ECTS Credits: Number of ECTS credits (can be "Not specified" if not available)
  * Local Credits: Number of local credits (can be "Not specified" if not available)
  * Semester: Fall, Spring, or Fall/Spring
  * Year: Academic year
  * Section: Which part of course info (course_metadata, objectives, description, prerequisites, nested_course, etc.)

FOR QUESTIONS ABOUT CREDITS/ECTS (e.g., "SE 115 kaç kredi", "How much ECTS is FR 103"):
  * ALWAYS check the "ECTS Credits" field in metadata - it shows the exact number or "Not specified"
  * ALWAYS check the "Local Credits" field in metadata
  * If ECTS Credits shows a number (e.g., "5"), answer: "SE 115 has 5 ECTS credits"
  * If asking "kaç kredi" (Turkish), provide BOTH ECTS and Local Credits: "SE 115 has 5 ECTS credits and 3 local credits"
  * If ECTS Credits shows "Not specified", check the Content field for credit information
  * DO NOT say "I don't have information" if ECTS Credits field exists - check if it's a number or "Not specified"

FOR QUESTIONS ABOUT MANDATORY/ELECTIVE COURSES:
  * Look for "Type: Mandatory" or "Type: Elective" in the Type field
  * For "List all mandatory courses in Software Engineering":
    - Find ALL sources where Department is "Software Engineering" AND Type is "Mandatory"
    - List EVERY course with: Course Code, Course Name, ECTS Credits
    - Format: "1. SE 115 - Introduction to Programming I (ECTS: 5, Type: Mandatory)"
  * DO NOT miss any courses - check ALL sources

FOR COURSE CODE QUESTIONS (e.g., "SE 115", "FR 103", "se 216", "fr103"):
  * Match course codes case-insensitively: "SE 115" = "se 115" = "SE115" = "se115"
  * "FR 103" = "fr103" = "FR103" = "fr 103" (all refer to the same course)
  * Look for the course code in the "Course Code" metadata field
  * Also check if course code appears in the Content text
  * For nested courses (FR 103, ITL 103, etc.), they may be in sources with section "nested_course"

FOR QUESTIONS ABOUT SPECIFIC COURSES:
  * Find the course by matching the course code in metadata (case-insensitive)
  * Look through ALL sources to find ALL matching course codes
  * Provide ALL available information: Course Code, Course Name, ECTS Credits, Local Credits, Type, Semester, Department
  * Example answer: "Course Code: SE 115, Course Name: Introduction to Programming I, ECTS Credits: 5, Local Credits: 3, Type: Mandatory, Department: Software Engineering"

FOR LIST QUESTIONS:
  * Provide COMPLETE lists with ALL matching courses from ALL sources
  * Include for each course: Course Code, Course Name, ECTS Credits, Local Credits, Type
  * Format: "1. SE 115 - Introduction to Programming I (ECTS: 5, Local Credits: 3, Type: Mandatory, Department: Software Engineering)"
  * DO NOT miss any courses - be thorough

IF INFORMATION IS NOT AVAILABLE:
  * Only say "I don't have information about this in the course data" if you've checked ALL sources and the information is truly not there
  * DO NOT say "I cannot provide guidance" or "Sorry, but I am unable to complete your request"
  * DO NOT make up information

BE SPECIFIC AND CITE:
  * Always include Course Code (e.g., "SE 115", "FR 103")
  * Always include Department name
  * Always include ECTS Credits and Local Credits (if available)
  * Always include Type (Mandatory/Elective)
  * Include any other relevant details from metadata

Answer:"""
        
        if self.llm_provider == "ollama":
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                )
                return response['response']
            except Exception as e:
                logger.error(f"Error generating response with Ollama: {e}")
                return f"Error generating response: {str(e)}"
        
        elif self.llm_provider == "huggingface":
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
                
                with torch.no_grad():
                    outputs = self.llm.generate(
                        inputs,
                        max_length=512,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove prompt from response
                response = response[len(prompt):].strip()
                return response
            except Exception as e:
                logger.error(f"Error generating response with HuggingFace: {e}")
                return f"Error generating response: {str(e)}"
    
    def query(self, query: str, n_results: int = 15,
              department_filter: Optional[str] = None,
              course_type_filter: Optional[str] = None) -> Dict:
        """
        Complete RAG query: retrieve and generate
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (optional)
            
        Returns:
            Dictionary with query, retrieved chunks, context, and response
        """
        # Enhance query for better retrieval
        enhanced_query = self._enhance_query(query)
        
        # Auto-detect filters from query if not provided
        if not department_filter or not course_type_filter:
            detected = self._detect_query_filters(query)
            if not department_filter:
                department_filter = detected.get('department_filter')
            if not course_type_filter:
                course_type_filter = detected.get('course_type_filter')
        
        # Step 1: Retrieve
        # If course code detected, also search with just the course code
        course_code = self._extract_course_code(query)
        if course_code:
            # Search with course code for better matching
            code_query = course_code
            retrieved_chunks = self.retrieve(
                query=code_query,
                n_results=n_results * 2,  # Get more results when searching by code
                department_filter=department_filter,
                course_type_filter=course_type_filter
            )
            # Also try enhanced query
            enhanced_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results,
                department_filter=department_filter,
                course_type_filter=course_type_filter
            )
            # Combine and deduplicate
            all_chunks = retrieved_chunks + enhanced_chunks
            seen_ids = set()
            unique_chunks = []
            for chunk in all_chunks:
                chunk_id = chunk.get('id', '')
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    unique_chunks.append(chunk)
            retrieved_chunks = unique_chunks
            # Prioritize by course code
            retrieved_chunks = self._prioritize_by_course_code(retrieved_chunks, course_code)
            # Limit to n_results
            retrieved_chunks = retrieved_chunks[:n_results]
        else:
            retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results,
                department_filter=department_filter,
                course_type_filter=course_type_filter
            )
        
        # Step 2: Generate context
        context = self.generate_context(retrieved_chunks)
        
        # Step 3: Generate response
        response = self.generate_response(query, context)
        
        return {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'context': context,
            'response': response,
            'num_results': len(retrieved_chunks)
        }
    
    def compare_departments(self, query: str, departments: List[str], n_results_per_dept: int = 3) -> Dict:
        """
        Compare information across multiple departments
        
        Args:
            query: User query
            departments: List of department names to compare
            n_results_per_dept: Number of results per department
            
        Returns:
            Dictionary with comparison results
        """
        all_results = {}
        
        for dept in departments:
            retrieved = self.retrieve(
                query=query,
                n_results=n_results_per_dept,
                department_filter=dept
            )
            all_results[dept] = retrieved
        
        # Generate comparison context
        comparison_context = "Comparison across departments:\n\n"
        for dept, chunks in all_results.items():
            comparison_context += f"=== {dept} ===\n"
            context = self.generate_context(chunks)
            comparison_context += context + "\n\n"
        
        # Generate comparison response
        comparison_query = f"Compare the following information across departments: {query}"
        response = self.generate_response(comparison_query, comparison_context)
        
        return {
            'query': query,
            'departments': departments,
            'results_by_department': all_results,
            'comparison_response': response
        }
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query with synonyms and variations"""
        enhanced = query
        
        # Turkish to English mappings
        turkish_terms = {
            'kaç kredi': 'ECTS credits local credits',
            'kredi': 'credits ECTS',
            'zorunlu': 'mandatory required',
            'seçmeli': 'elective',
            'ders': 'course',
            'bölüm': 'department',
            'yıl': 'year',
            'dönem': 'semester'
        }
        
        for turkish, english in turkish_terms.items():
            if turkish.lower() in query.lower():
                enhanced += f" {english}"
        
        return enhanced
    
    def _extract_course_code(self, query: str) -> Optional[str]:
        """Extract course code from query"""
        pattern = r'\b([A-Z]{2,4})\s*(\d{3,4})\b'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            code = match.group(1).upper()
            num = match.group(2)
            return f"{code} {num}"
        return None
    
    def _detect_query_filters(self, query: str) -> Dict[str, Optional[str]]:
        """Detect department and course type from query"""
        query_lower = query.lower()
        
        department_filter = None
        if 'software engineering' in query_lower:
            department_filter = 'Software Engineering'
        elif 'computer engineering' in query_lower:
            department_filter = 'Computer Engineering'
        elif 'electrical' in query_lower or 'electronics' in query_lower:
            department_filter = 'Electrical and Electronics Engineering'
        elif 'industrial engineering' in query_lower:
            department_filter = 'Industrial Engineering'
        
        course_type_filter = None
        if 'mandatory' in query_lower or 'required' in query_lower or 'zorunlu' in query_lower:
            course_type_filter = 'Mandatory'
        elif 'elective' in query_lower or 'seçmeli' in query_lower:
            course_type_filter = 'Elective'
        
        return {
            'department_filter': department_filter,
            'course_type_filter': course_type_filter
        }
    
    def _prioritize_by_course_code(self, chunks: List[Dict], course_code: str) -> List[Dict]:
        """Prioritize chunks with matching course code"""
        # Normalize course code for comparison (remove spaces, uppercase)
        normalized_code = course_code.replace(' ', '').upper()
        
        # Also try with space variations
        code_variations = [
            normalized_code,
            course_code.upper().replace(' ', ''),
            course_code.upper(),
            course_code.replace(' ', ''),
        ]
        
        exact_matching = []  # Exact match
        partial_matching = []  # Contains course code
        non_matching = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            chunk_code = metadata.get('course_code', '')
            normalized_chunk_code = chunk_code.replace(' ', '').upper()
            
            # Check for exact match in metadata
            if normalized_chunk_code in code_variations:
                exact_matching.append(chunk)
            # Check if course code appears in text (for nested courses or descriptions)
            elif any(var in chunk.get('text', '').upper() for var in code_variations):
                partial_matching.append(chunk)
            else:
                non_matching.append(chunk)
        
        # Return: exact matches first, then text matches, then others
        return exact_matching + partial_matching + non_matching
