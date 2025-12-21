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
from .answer_formatter import format_answer, build_no_data_answer, build_partial_answer, build_pool_course_answer

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
    
    def __init__(self, vector_db, embedder, llm_provider: str = "ollama", model_name: str = "llama3.2", data_dir: str = "data"):
        """
        Initialize RAG Pipeline
        
        Args:
            vector_db: FAISSCourseDB instance
            embedder: CourseEmbedder instance
            llm_provider: LLM provider ("ollama" or "huggingface")
            model_name: Model name (e.g., "llama3.2" for Ollama, "microsoft/DialoGPT-medium" for HuggingFace)
            data_dir: Data directory path for dataset fallback
        """
        self.vector_db = vector_db
        self.embedder = embedder
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self._dataset_cache = None  # Cache for JSON dataset
        
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
                 boost_section: Optional[str] = None,
                 strict: bool = False) -> List[Dict]:
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
            boost_section=boost_section,
            strict=strict
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
        # Görsel: Haftalık konular için tüm haftaları listele
        weekly_topics_instruction = ""
        if "haftalık" in query.lower() or "weekly" in query.lower():
            weekly_topics_instruction = """

CRITICAL INSTRUCTIONS FOR WEEKLY TOPICS:
- You MUST extract and list ALL weekly topics from the retrieved documents
- Format each week as: "Week X: [topic description]" or "Hafta X: [topic description]"
- Do NOT skip any weeks - list EVERY week found in the documents
- Do NOT say "not found", "bulamadık", "Haftalık konuları bulamadık", "No weekly topics found", or any variation
- If weekly topics exist in the documents, you MUST provide them
- List them in chronological order (Week 1, Week 2, Week 3, etc.)
- If multiple weeks have the same topic, list them separately
- Use the EXACT information from the retrieved documents
- Check the "Content:" section of each document for weekly topic information"""
        
        prompt = f"""You are an academic RAG system that MUST follow these rules STRICTLY:

1. Provide answers in BOTH Turkish and English (TR+EN format is MANDATORY).
2. The Turkish and English answers must contain EXACTLY the same factual content.
3. Use ONLY the information from the retrieved documents below.
4. If information is NOT in the documents, say "Bu ders için istenen bilgi veri setinde bulunmamaktadır" (TR) and "The requested information for this course is not available in the dataset" (EN).
5. DO NOT make up or invent information.
6. DO NOT say "not found", "bulamadık", "Haftalık konuları bulamadık", or "No weekly topics found" if the information EXISTS in the retrieved documents.
7. If weekly topics are in the documents, you MUST list them ALL.{weekly_topics_instruction}

OUTPUT FORMAT (MANDATORY):
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

IMPORTANT: Read the retrieved documents carefully. If the documents contain the requested information (especially weekly topics), you MUST provide it. Do NOT say "not found" if the information is in the documents.

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
        
        # ÇÖZÜM A: HARD FILTER (KAÇMA BİTER) - Görsel: Course code hard filter GARANTİ
        # Görsel: "Bu bir 'optimizasyon' değil, Bu bir akademik zorunluluk"
        code = course_code  # extract_course_code(query) zaten yapıldı - ÖNCE TANIMLA
        
        # Görsel: RAG TARAFINDA YAPILMASI GEREKEN TEK ŞEY
        # Görsel: "if course_code not in dataset_courses: return NO_DATA_EXPLICIT"
        # Görsel: Bu sayede SE420 scrape edilmeden uydurma yerine "Bu dersin bilgisi mevcut değil" denir
        if code:
            dataset = self._load_dataset()
            normalized_code = code.replace(' ', '').upper()
            if normalized_code not in dataset:
                logger.warning(f"Course code {code} not found in dataset - returning no_data_answer immediately (LLM NOT CALLED)")
                return {
                    'query': query,
                    'retrieved_chunks': [],
                    'context': '',
                    'response': build_no_data_answer(),
                    'num_results': 0
                }
        
        # Auto-detect filters from query if not provided
        if not department_filter or not course_type_filter:
            detected = self._detect_query_filters(query)
            if not department_filter:
                department_filter = detected.get('department_filter')
            if not course_type_filter:
                course_type_filter = detected.get('course_type_filter')
        
        # Görsel: Department filter zorunlu - Core courses gibi sorularda
        # Görsel: "Retriever'a şu filtre zorunlu olmalı: department == 'Software Engineering'"
        # Görsel: "Bu filtre şu an her soruda aktif değil" → Zorunlu yap
        query_lower = query.lower()
        
        # Görsel (2): Section-aware retrieval hâlâ "soft"
        # Görsel: "sadece boost ediliyorsa ❌" ve "ama zorlanmıyorsa ❌"
        # Görsel: "Haftalık konu sorusu, objectives'tan cevaplanabilir" → Hard filter olmalı
        section = None  # ÖNCE TANIMLA
        requested_section = None  # Görsel: Fallback için hangi section arandığını hatırla
        if "kredi" in query_lower or "credit" in query_lower or "ects" in query_lower:
            section = "credits"  # Görsel: "prerequisites" değil, "credits" olmalı
            requested_section = "credits"
        elif "haftalık" in query_lower or "weekly" in query_lower:
            section = "weekly_topics"  # Görsel: "objectives" değil, "weekly_topics" olmalı
            requested_section = "weekly_topics"
        
        intent_detected = None
        if not code:  # Course code yoksa intent kontrolü yap
            if "core" in query_lower or "important" in query_lower or "önemli" in query_lower or "temel" in query_lower:
                intent_detected = "core_courses"  # Görsel: "important_courses", "core_courses"
            elif "mandatory" in query_lower or "required" in query_lower or "zorunlu" in query_lower:
                intent_detected = "mandatory_courses"
            
            # Görsel: "if no_course_code and intent in ['important_courses', 'core_courses']: restrict retrieval to: department == 'Software Engineering' AND section in ['description', 'objectives']"
            if intent_detected in ["core_courses", "important_courses", "mandatory_courses"]:
                # Department filter zorunlu (Görsel: Akademik hata önleme)
                if not department_filter:
                    # Query'den department çıkar, yoksa default Software Engineering
                    if "software engineering" in query_lower or "yazılım mühendisliği" in query_lower:
                        department_filter = "Software Engineering"
                    elif "computer engineering" in query_lower or "bilgisayar mühendisliği" in query_lower:
                        department_filter = "Computer Engineering"
                    elif "electrical" in query_lower or "electronics" in query_lower:
                        department_filter = "Electrical and Electronics Engineering"
                    elif "industrial engineering" in query_lower:
                        department_filter = "Industrial Engineering"
                    else:
                        # Default: Software Engineering (en çok sorulan)
                        department_filter = "Software Engineering"
                        logger.info(f"No department specified in query, defaulting to Software Engineering for intent: {intent_detected}")
                
                # Görsel: Section filter - "section in ['description', 'objectives']"
                # Core courses için sadece description ve objectives section'larından arama yap
                if not section:  # Eğer başka bir section filter yoksa
                    # Core courses için section filter ekle (description veya objectives)
                    # Bu filter'ı retrieve'de kullanacağız
                    logger.info(f"Intent detected: {intent_detected}, enforcing department filter: {department_filter}, restricting to sections: description, objectives")
        
        # Görsel: Pool course kontrolü (ELEC 001, POOL 003 gibi)
        # Görsel: "Bu bilgi dataset'te dolaylı olarak var, ama sen pipeline'da pool logic'i hiç ele almıyorsun"
        is_pool_course = False
        if code:
            code_upper = code.upper()
            if code_upper.startswith("ELEC") or code_upper.startswith("POOL") or code_upper.startswith("SFL"):
                is_pool_course = True
                # Görsel: Pool course için özel açıklama
                if requested_section == "credits":
                    logger.info(f"Pool course detected: {code}, returning pool course explanation")
                    return {
                        'query': query,
                        'retrieved_chunks': [],
                        'context': '',
                        'response': build_pool_course_answer(code),
                        'num_results': 0
                    }
        
        # Görsel: Section varsa, filters'a ekle (hard filter, sadece boost değil)
        if section and code:
            # Course code + section hard filter
            filters = {"course_code": code, "section": section}
        elif code:
            # Sadece course code filter
            filters = {"course_code": code}
        elif section:
            # Sadece section filter (hard filter)
            filters = {"section": section}
        else:
            filters = None
        
        # Görsel: ÇÖZÜM - Exact metadata lookup önce, sonra similarity
        # Görsel: "1) Metadata ile önce daralt, sonra similarity"
        # Görsel: "Exact metadata lookup (course_code+section) → önce"
        mark_as_partial = False  # Görsel: Fallback mekanizması için flag
        retrieved_chunks = []
        
        # Görsel: ÖNCE - Dataset fallback kontrolü (eğer exact lookup başarısız olursa kullanılacak)
        # Görsel: Kredi bilgisi için önce dataset'te kontrol et
        dataset_value = None
        if code and requested_section:
            dataset_value = self._dataset_lookup(course_code=code, section=requested_section)
            if dataset_value:
                logger.info(f"Dataset has exact data for {code}, section={requested_section}")
        
        # Görsel: ÖNCE - Exact metadata lookup (course_code + section)
        if filters and code:
            # Görsel: "docs = db.search_by_metadata({ 'course_code': course_code, 'section': section })"
            # Görsel: Haftalık konular için tüm chunk'ları al (her hafta ayrı chunk)
            lookup_n_results = n_results * 20 if requested_section == "weekly_topics" else n_results
            exact_chunks = self.vector_db.search_by_metadata(filters, n_results=lookup_n_results)
            if exact_chunks:
                logger.info(f"Exact metadata lookup found {len(exact_chunks)} chunks for {filters}")
                retrieved_chunks = exact_chunks
                # Görsel: Eğer exact lookup başarılı olduysa, dataset'te de varsa partial answer ekleme
                if dataset_value:
                    logger.info(f"Exact lookup successful, dataset also has data - using exact lookup results")
            else:
                # Görsel: "if not docs: docs = db.search_by_metadata({ 'course_code': course_code })"
                # Fallback: Sadece course_code ile arama (section filter'ı kaldır)
                if requested_section:
                    logger.info(f"Exact section '{requested_section}' not found for {code}, trying fallback: course_code only")
                    fallback_filters = {"course_code": code}
                    # Görsel: Haftalık konular için tüm chunk'ları al
                    fallback_n_results = n_results * 30 if requested_section == "weekly_topics" else n_results * 2
                    exact_chunks = self.vector_db.search_by_metadata(fallback_filters, n_results=fallback_n_results)
                    if exact_chunks:
                        # Görsel: Fallback'te section'a göre filtrele (weekly_topics için tüm haftaları al)
                        if requested_section == "weekly_topics":
                            # Sadece weekly_topics section'ından chunk'ları al (tüm haftalar)
                            exact_chunks = [chunk for chunk in exact_chunks 
                                          if chunk.get('metadata', {}).get('section', '').lower() == 'weekly_topics']
                        retrieved_chunks = exact_chunks
                        mark_as_partial = True
                        logger.info(f"Fallback successful: found {len(retrieved_chunks)} chunks for {code}")
        
        # Görsel: SONRA - Similarity search (eğer exact lookup sonuç vermezse)
        if not retrieved_chunks:
            # Görsel: "Similarity search → sonra"
            retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results * 2 if filters else n_results,
                department_filter=department_filter,
                course_type_filter=course_type_filter,
                filters=filters,  # Görsel: Hard filter GARANTİ
                boost_section=section,
                strict=True if filters else False
            )
        
        # Eğer hala chunk yoksa, normal search yap (intent detection için)
        if not retrieved_chunks:
            # Course code ve section yoksa normal search
            # Görsel: General query guard - Course code yoksa ve intent belirsizse, çok sıkı filtreleme
            # Görsel: "if retrieved_chunks < MIN_THRESHOLD: DO NOT call LLM, return 'Dataset-based answer not available'"
            if not code and intent_detected in ["core_courses", "important_courses", "mandatory_courses"]:
                # Görsel: Core courses için section filter ekle
                # Görsel: "restrict retrieval to: department == 'Software Engineering' AND section in ['description', 'objectives']"
                # Görsel: Core courses için course_type='Mandatory' filter ekle ve genel dersleri (TURK, IUE, HIST) filtrele
                retrieved_chunks = self.retrieve(
                    query=enhanced_query,
                    n_results=n_results * 3,  # Daha fazla sonuç al, sonra filtrele
                    department_filter=department_filter,  # Görsel: Department filter zorunlu
                    course_type_filter='Mandatory',  # Görsel: Sadece Mandatory courses (core courses)
                    filters=None,
                    boost_section=None,
                    strict=False
                )
                # Görsel: Section filter - sadece description ve objectives
                # Görsel: Genel dersleri filtrele (TURK, IUE, HIST, ENG gibi genel eğitim dersleri)
                if retrieved_chunks:
                    filtered_chunks = []
                    excluded_codes = ['TURK', 'IUE', 'HIST', 'ENG101', 'ENG102', 'ENG210', 'TURK100', 'IUE100', 'HIST100']  # Genel eğitim dersleri
                    for chunk in retrieved_chunks:
                        chunk_metadata = chunk.get('metadata', {})
                        chunk_section = chunk_metadata.get('section', '').lower()
                        chunk_code = chunk_metadata.get('course_code', '').upper()
                        
                        # Section filter: description veya objectives
                        if chunk_section in ['description', 'objectives']:
                            # Genel dersleri filtrele (TURK, IUE, HIST, ENG ile başlayanlar)
                            is_general_course = any(chunk_code.startswith(excluded) for excluded in excluded_codes)
                            if not is_general_course:
                                filtered_chunks.append(chunk)
                    
                    if filtered_chunks:
                        retrieved_chunks = filtered_chunks[:n_results]  # Top n_results
                        logger.info(f"Section filter applied: {len(filtered_chunks)} chunks from description/objectives sections (excluded general courses: {excluded_codes})")
                    else:
                        # Eğer description/objectives yoksa, tüm section'ları kullan (fallback)
                        logger.warning(f"No chunks found in description/objectives sections, using all sections")
            elif not code and not intent_detected:
                # Görsel: Intent belirsiz, course code yok → Çok sıkı filtreleme
                # Department filter zorunlu (eğer belirtilmişse)
                if department_filter:
                    # Department filter ile arama yap
                    retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results,
                department_filter=department_filter,
                        course_type_filter=course_type_filter,
                        filters=None,
                        boost_section=section,
                        strict=False
                    )
                else:
                    # Görsel: "Retriever her şeyi getiriyor" → Çok geniş arama, guard gerekli
                    retrieved_chunks = self.retrieve(
                        query=enhanced_query,
                        n_results=n_results,
                        department_filter=None,
                        course_type_filter=course_type_filter,
                        filters=None,
                        boost_section=section,
                        strict=False
                    )
        else:
                # Normal search (code var ama filters yok)
            retrieved_chunks = self.retrieve(
                query=enhanced_query,
                n_results=n_results,
                department_filter=department_filter,
                    course_type_filter=course_type_filter,
                    filters=None,
                    boost_section=section,
                    strict=False
                )
        
        # Görsel (3): "Veri yok" guard hâlâ %100 değil
        # Görsel: "az context ile tahmin yapmaya devam eder" → Bu da hallucination kapısıdır
        max_similarity = max((d.get('similarity', 0) for d in retrieved_chunks), default=0) if retrieved_chunks else 0
        min_chunks_for_context = 3  # Görsel: Az context ile LLM tahmin yapmaya devam eder
        
        # Görsel: General query guard - "if retrieved_chunks < MIN_THRESHOLD: DO NOT call LLM"
        # Görsel: "Bu satır hallucination'ı %90 keser"
        MIN_THRESHOLD = 5  # Görsel: Minimum chunk sayısı
        if not code and not intent_detected:
            # Görsel: Course code yok, intent belirsiz → Çok sıkı guard
            if len(retrieved_chunks) < MIN_THRESHOLD:
                logger.warning(f"General query guard triggered: chunks={len(retrieved_chunks)} < MIN_THRESHOLD={MIN_THRESHOLD}, intent={intent_detected}")
                tr = "Bu soru için yeterli veri bulunamadı. Lütfen daha spesifik bir soru sorun (örneğin, bir ders kodu belirtin)."
                en = "Insufficient data found for this query. Please ask a more specific question (e.g., specify a course code)."
                return {
                    'query': query,
                    'retrieved_chunks': retrieved_chunks,
                    'context': '',
                    'response': format_answer(tr, en),
                    'num_results': len(retrieved_chunks)
                }
        
        # Görsel: Guard logic - Dataset fallback kontrolü
        # Görsel: "3) Guard (Veri yok) sadece gerçekten hiçbir şey yoksa"
        # Görsel: "if not docs or max_similarity < MIN_SIM: if dataset_has(course_code, section): use dataset value directly"
        if not retrieved_chunks or max_similarity < 0.30 or len(retrieved_chunks) < min_chunks_for_context:
            # Görsel: Eğer JSON'da **credits** var ama retriever çağırmadıysa
            # Görsel: ÖNCE dataset fallback kontrolü yap (guard'dan önce)
            if code and requested_section:
                dataset_value = self._dataset_lookup(course_code=code, section=requested_section)
                if dataset_value:
                    logger.info(f"Dataset fallback found data for {code}, section={requested_section}")
                    # Dataset'ten direkt değer kullan (partial answer değil, direkt cevap)
                    tr_content = dataset_value.get('tr', f"{code} dersinin {requested_section} bilgisi: {dataset_value.get('value', '')}")
                    en_content = dataset_value.get('en', f"{code} course {requested_section} information: {dataset_value.get('value', '')}")
                    return {
                        'query': query,
                        'retrieved_chunks': [],
                        'context': '',
                        'response': format_answer(tr_content, en_content),
                        'num_results': 0
                    }
            
            # Görsel: Similarity düşükse VEYA az context varsa LLM'e gitmeden kesin guard
            logger.warning(f"Guard triggered: chunks={len(retrieved_chunks)}, max_similarity={max_similarity:.3f}")
            return {
                'query': query,
                'retrieved_chunks': [],
                'context': '',
                'response': build_no_data_answer(),
                'num_results': 0
            }
        
        # Step 2: Generate context
        context = self.generate_context(retrieved_chunks)
        
        # Görsel: Context çok kısa ise (az context) yine guard
        # Görsel: Ama eğer weekly_topics sorgusuysa ve chunks varsa, context kısa olsa bile devam et
        is_weekly_topics_query = "haftalık" in query.lower() or "weekly" in query.lower()
        # Görsel: Weekly topics sorgusu için chunks varsa context kısa olsa bile devam et
        if len(context.strip()) < 100 and not (is_weekly_topics_query and retrieved_chunks):  # Çok kısa context (weekly topics + chunks varsa hariç)
            logger.warning(f"Context too short ({len(context)} chars), returning no_data_answer")
            return {
                'query': query,
                'retrieved_chunks': retrieved_chunks,
                'context': context,
                'response': build_no_data_answer(),
                'num_results': len(retrieved_chunks)
            }
        
        # Step 3: Generate response
        response = self.generate_response(query, context)
        
        # Görsel: Fallback mekanizması - Partial answer işaretleme
        # Görsel: "Ve cevapta şunu ekle: Bu ders için kredi bilgisi veri setinde doğrudan bulunamadığı için..."
        # Görsel: Ama eğer dataset'te varsa partial answer ekleme (çünkü zaten dataset fallback'te direkt cevap verdik)
        if mark_as_partial and requested_section:
            # Dataset'te kontrol et - eğer varsa partial answer ekleme
            if not dataset_value:  # Yukarıda zaten kontrol edildi
                dataset_value = self._dataset_lookup(course_code=code, section=requested_section)
            
            if not dataset_value:
                # Dataset'te yok, partial answer ekle
                tr_match = re.search(r'(?:ANSWER|CEVAP)\s*\(TR\)[:\-]*\s*(.*?)(?=(?:ANSWER|CEVAP)\s*\(EN\)|$)', response, re.IGNORECASE | re.DOTALL)
                en_match = re.search(r'(?:ANSWER|CEVAP)\s*\(EN\)[:\-]*\s*(.*?)$', response, re.IGNORECASE | re.DOTALL)
                
                if tr_match and en_match:
                    tr_content = tr_match.group(1).strip()
                    en_content = en_match.group(1).strip()
                    response = build_partial_answer(tr_content, en_content, info_type="kredi" if requested_section == "credits" else requested_section)
                    logger.info(f"Marked answer as partial for {code} (requested section: {requested_section})")
            else:
                # Dataset'te var, partial answer ekleme (zaten exact bilgi var)
                logger.info(f"Dataset has exact data for {code}, section={requested_section}, not marking as partial")
        
        # Görsel: TR+EN format assertion (Kod seviyesinde garanti)
        # Görsel: "assert 'ANSWER (TR)' in answer" ve "assert 'ANSWER (EN)' in answer"
        # Görsel: "LLM 'unutursa' sistem çöker" → Kod seviyesinde kontrol
        # Görsel: "Prompt'a güvenmek yeterli değil" → Kod seviyesinde garanti
        response_upper = response.upper()
        
        # Görsel: Kod seviyesinde assertion (main.py'de değil, burada)
        has_tr = "ANSWER (TR)" in response_upper or "CEVAP (TR)" in response_upper
        has_en = "ANSWER (EN)" in response_upper or "CEVAP (EN)" in response_upper
        
        if not has_tr or not has_en:
            # LLM format'ı unuttuysa, format_answer ile zorla
            logger.warning("LLM response does not contain TR+EN format, enforcing format...")
            # Try to extract TR and EN from response
            tr_match = re.search(r'(?:ANSWER|CEVAP)\s*\(TR\)[:\-]*\s*(.*?)(?=(?:ANSWER|CEVAP)\s*\(EN\)|$)', response, re.IGNORECASE | re.DOTALL)
            en_match = re.search(r'(?:ANSWER|CEVAP)\s*\(EN\)[:\-]*\s*(.*?)$', response, re.IGNORECASE | re.DOTALL)
            
            if tr_match and en_match:
                tr_text = tr_match.group(1).strip()
                en_text = en_match.group(1).strip()
                response = format_answer(tr_text, en_text)
            else:
                # Fallback: Use same text for both (LLM tek dil döndüyse)
                logger.warning("LLM returned single language, duplicating for TR+EN format")
                response = format_answer(response, response)
        
        # Görsel: Final assertion (kod seviyesinde garanti)
        # Görsel: "LLM bir gün tek dil döner ve sistem 'sessizce bozulur'" → Assert ile önle
        # Görsel: "if 'ANSWER (TR)' not in answer or 'ANSWER (EN)' not in answer: raise RuntimeError('Bilingual output violation')"
        final_check = response.upper()
        if "ANSWER (TR)" not in final_check and "CEVAP (TR)" not in final_check:
            raise RuntimeError("Bilingual output violation: TR section missing in response - system guarantee failed")
        if "ANSWER (EN)" not in final_check and "CEVAP (EN)" not in final_check:
            raise RuntimeError("Bilingual output violation: EN section missing in response - system guarantee failed")
        
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
    
    def _load_dataset(self) -> Dict:
        """
        Görsel: Dataset fallback - JSON'dan direkt okuma
        Görsel: "Dataset fallback → sadece gerçekten yoksa"
        
        Returns:
            Dictionary with course data from JSON files
        """
        if self._dataset_cache is not None:
            return self._dataset_cache
        
        dataset = {}
        raw_data_path = self.data_dir / 'raw' / 'scraped_courses.json'
        
        if raw_data_path.exists():
            try:
                with open(raw_data_path, 'r', encoding='utf-8') as f:
                    all_data = json.load(f)
                
                # Flatten department structure
                # Görsel: Nested course'ları da ekle (parent_course field'ı olan course'lar)
                for dept_key, courses in all_data.items():
                    for course in courses:
                        course_code = course.get('course_code', '')
                        if not course_code:
                            continue
                        # Normalize course code (remove spaces, uppercase)
                        normalized_code = course_code.replace(' ', '').upper()
                        if normalized_code:
                            # Görsel: Nested course'ları da dataset'e ekle (SE420, GER101, GEET312 gibi)
                            dataset[normalized_code] = course
                        
                        # Görsel: available_courses içindeki nested course'ları da ayrı course olarak ekle
                        # (Eğer zaten ayrı course olarak eklenmemişse)
                        available_courses = course.get('available_courses', [])
                        for nested in available_courses:
                            nested_code = nested.get('course_code', '')
                            if nested_code:
                                nested_normalized = nested_code.replace(' ', '').upper()
                                # Eğer nested course zaten dataset'te yoksa ekle
                                if nested_normalized and nested_normalized not in dataset:
                                    # Nested course'u ayrı course olarak ekle
                                    nested_as_course = {
                                        'course_code': nested_code,
                                        'course_name': nested.get('course_name', ''),
                                        'detail_url': nested.get('detail_url', ''),
                                        'semester': nested.get('semester', ''),
                                        'type': nested.get('type', 'Elective'),
                                        'ects': nested.get('ects'),
                                        'local_credits': nested.get('local_credits'),
                                        'objectives': nested.get('objectives', ''),
                                        'description': nested.get('description', ''),
                                        'weekly_topics': nested.get('weekly_topics', []),
                                        'learning_outcomes': nested.get('learning_outcomes', []),
                                        'assessment': nested.get('assessment', {}),
                                        'ects_workload': nested.get('ects_workload', {}),
                                        'prerequisites': nested.get('prerequisites', ''),
                                        'department': course.get('department', ''),
                                        'department_key': course.get('department_key', dept_key),
                                        'parent_course': course_code  # Track parent
                                    }
                                    dataset[nested_normalized] = nested_as_course
                                    logger.debug(f"Added nested course {nested_code} to dataset from available_courses")
                
                self._dataset_cache = dataset
                logger.info(f"Loaded {len(dataset)} courses from dataset")
            except Exception as e:
                logger.warning(f"Error loading dataset: {e}")
                self._dataset_cache = {}
        else:
            logger.warning(f"Dataset file not found: {raw_data_path}")
            self._dataset_cache = {}
        
        return self._dataset_cache
    
    def _dataset_lookup(self, course_code: str, section: str) -> Optional[Dict]:
        """
        Görsel: Dataset lookup - JSON'dan direkt değer okuma
        Görsel: "dataset_lookup(course_code='SE115', section='credits') -> found"
        
        Args:
            course_code: Normalized course code (e.g., "SE115")
            section: Section name (e.g., "credits", "weekly_topics")
            
        Returns:
            Dictionary with value and TR+EN content, or None if not found
        """
        dataset = self._load_dataset()
        normalized_code = course_code.replace(' ', '').upper()
        
        if normalized_code not in dataset:
            return None
        
        course = dataset[normalized_code]
        
        # Görsel: Section'a göre değer bul
        if section == "credits":
            ects = course.get('ects')
            local_credits = course.get('local_credits')
            if ects is not None or local_credits is not None:
                value = f"ECTS: {ects}, Local Credits: {local_credits}" if ects and local_credits else f"ECTS: {ects}" if ects else f"Local Credits: {local_credits}"
                return {
                    'value': value,
                    'tr': f"{course_code} dersinin kredisi: {value}",
                    'en': f"{course_code} course credits: {value}"
                }
        elif section == "weekly_topics":
            weekly_topics = course.get('weekly_topics', [])
            if weekly_topics:
                # Görsel: Tüm haftaları formatla
                topics_list = []
                for t in weekly_topics:
                    week = t.get('week', '')
                    topic = t.get('topic', '')
                    required_materials = t.get('required_materials', '')
                    if week and topic:
                        topic_line = f"Hafta {week}: {topic}"
                        if required_materials:
                            topic_line += f"\nGerekli Materyaller: {required_materials}"
                        topics_list.append(topic_line)
                
                topics_text = "\n".join(topics_list)
                return {
                    'value': topics_text,
                    'tr': f"{course_code} dersinin haftalık konuları:\n{topics_text}",
                    'en': f"{course_code} course weekly topics:\n{topics_text}"
                }
        
        return None
    
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
