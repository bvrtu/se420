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
from .answer_formatter import format_answer, build_no_data_answer

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
                 course_type_filter: Optional[str] = None,
                 filters: Optional[Dict] = None,
                 boost_section: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (optional)
            filters: Dictionary with metadata filters (course_code, department, section) - STRING EQUALITY
            boost_section: Section to boost (e.g., 'credits', 'weekly_topics')
            
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
            course_type_filter=course_type_filter,
            filters=filters,
            boost_section=boost_section
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
        # KESİN PROMPT (BİREBİR KULLAN) - Görseldeki template'e göre güncellendi
        prompt = f"""You are an academic RAG system that MUST follow these rules:

1. Provide answers in BOTH Turkish and English.
2. The Turkish and English answers must contain EXACTLY the same factual content.
3. Use ONLY the retrieved documents.
4. If info is missing, say so clearly in both languages.

OUTPUT FORMAT:
------------------------------------------------------------
ANSWER (TR)
------------------------------------------------------------
{{turkish_answer}}
------------------------------------------------------------
ANSWER (EN)
------------------------------------------------------------
{{english_answer}}
------------------------------------------------------------

Retrieved Documents:
{context}

User Question: {query}

Answer:"""
        
        if self.llm_provider == "ollama":
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.3,  # Lower temperature for more factual responses
                        "top_p": 0.9,
                    }
                )
                raw_response = response['response']
                
                # Format response through answer_formatter (ÇÖZÜM: LLM'den gelen cevabı buradan geçirmeden yazdırma)
                # Check if response already has TR/EN format
                if "ANSWER (TR)" in raw_response and "ANSWER (EN)" in raw_response:
                    return raw_response
                else:
                    # If LLM didn't follow format, create TR+EN format
                    # Try to detect if response is in Turkish or English
                    tr_response = raw_response
                    en_response = raw_response  # Default: same for both
                    return format_answer(tr_response, en_response)
            except Exception as e:
                logger.error(f"Error generating response with Ollama: {e}")
                return build_no_data_answer()
        
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
              course_type_filter: Optional[str] = None,
              course_code: Optional[str] = None) -> Dict:
        """
        Complete RAG query: retrieve and generate
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            department_filter: Filter by department (optional)
            course_type_filter: Filter by course type (optional)
            course_code: Extracted course code from query (optional, will extract if not provided)
            
        Returns:
            Dictionary with query, retrieved chunks, context, and response
        """
        # Enhance query for better retrieval
        enhanced_query = self._enhance_query(query)
        
        # Auto-detect course code if not provided
        if not course_code:
            course_code = self._extract_course_code(query)
        
        # Auto-detect filters from query if not provided
        if not department_filter or not course_type_filter:
            detected = self._detect_query_filters(query)
            if not department_filter:
                department_filter = detected.get('department_filter')
            if not course_type_filter:
                course_type_filter = detected.get('course_type_filter')
        
        # ÇÖZÜM A: HARD FILTER (KAÇMA BİTER) - Görsel: Course code hard filter GARANTİ
        # Görsel: "Bu bir 'optimizasyon' değil, Bu bir akademik zorunluluk"
        code = course_code  # extract_course_code(query) zaten yapıldı
        
        # ÇÖZÜM B: SECTION-AWARE RETRIEVAL - Görsel: Section mapping düzeltmesi
        # Görsel: "haftalık" → "objectives" YANLIŞ, "weekly_topics" DOĞRU
        # Görsel: "kredi" → "prerequisites" YANLIŞ, "credits" DOĞRU
        section = None
        query_lower = query.lower()
        if "kredi" in query_lower or "credit" in query_lower or "ects" in query_lower:
            section = "credits"  # Görsel: "prerequisites" değil, "credits" olmalı
        elif "haftalık" in query_lower or "weekly" in query_lower:
            section = "weekly_topics"  # Görsel: "objectives" değil, "weekly_topics" olmalı
        
        # Step 1: Retrieve with HARD FILTER (Görsel: Garanti olmalı)
        # Görsel: "if course_code: filters={"course_code": course_code}" zorunlu
        if code:
            # Görsel: Course code varsa MUTLAKA filter uygula (akademik zorunluluk)
            filters = {"course_code": code}
            retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results * 2,
                department_filter=department_filter,
                course_type_filter=course_type_filter,
                filters=filters,  # Görsel: Hard filter GARANTİ
                boost_section=section
            )
        else:
            # Course code yoksa normal search
            retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results,
                department_filter=department_filter,
                course_type_filter=course_type_filter,
                filters=None,
                boost_section=section
            )
        
        # ÇÖZÜM C: DATA-NOT-FOUND GUARD - Görsel: "Veri yok" guard hâlâ zayıf
        # Görsel: "Benzerlik düşük ama LLM cevap üretiyor" → Kod seviyesinde kesin guard
        max_similarity = max((d.get('similarity', 0) for d in retrieved_chunks), default=0) if retrieved_chunks else 0
        if not retrieved_chunks or max_similarity < 0.30:
            # Görsel: Similarity düşükse LLM'e gitmeden kesin guard
            return {
                'query': query,
                'retrieved_chunks': [],
                'context': '',
                'response': build_no_data_answer(),
                'num_results': 0
            }
        
        # Step 2: Generate context
        context = self.generate_context(retrieved_chunks)
        
        # Step 3: Generate response
        response = self.generate_response(query, context)
        
        # Görsel: TR+EN format assertion (Kod seviyesinde garanti)
        # Görsel: "assert 'ANSWER (TR)' in answer" ve "assert 'ANSWER (EN)' in answer"
        # Görsel: "LLM 'unutursa' sistem çöker" → Kod seviyesinde kontrol
        response_upper = response.upper()
        if "ANSWER (TR)" not in response_upper or "ANSWER (EN)" not in response_upper:
            # LLM format'ı unuttuysa, format_answer ile zorla
            logger.warning("LLM response does not contain TR+EN format, enforcing format...")
            # Try to extract TR and EN from response
            tr_match = re.search(r'ANSWER\s*\(TR\)[:\-]*\s*(.*?)(?=ANSWER\s*\(EN\)|$)', response, re.IGNORECASE | re.DOTALL)
            en_match = re.search(r'ANSWER\s*\(EN\)[:\-]*\s*(.*?)$', response, re.IGNORECASE | re.DOTALL)
            
            if tr_match and en_match:
                tr_text = tr_match.group(1).strip()
                en_text = en_match.group(1).strip()
                response = format_answer(tr_text, en_text)
            else:
                # Fallback: Use same text for both
                response = format_answer(response, response)
        
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
