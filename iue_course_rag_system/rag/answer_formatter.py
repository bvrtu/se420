"""
Answer Formatter Module
Formats RAG responses in both Turkish and English
Prevents hallucination by providing structured "no data" responses
"""


def format_answer(tr: str, en: str) -> str:
    """
    Format answer in both Turkish and English (ÇÖZÜM: TR+EN cevap yapısı)
    
    Args:
        tr: Turkish answer
        en: English answer
        
    Returns:
        Formatted answer string
    """
    return f"""------------------------------------------------------------
ANSWER (TR)
------------------------------------------------------------
{tr}
------------------------------------------------------------
ANSWER (EN)
------------------------------------------------------------
{en}
------------------------------------------------------------"""


def build_no_data_answer() -> str:
    """
    Build standardized "no data" answer (ÇÖZÜM: "VERİ YOK" CEVABI - HALLUCINATION BİTER)
    
    Returns:
        Formatted "no data" answer in both languages
    """
    tr = "Bu ders için istenen bilgi veri setinde bulunmamaktadır."
    en = "The requested information for this course is not available in the dataset."
    return format_answer(tr, en)


def build_partial_answer(tr_content: str, en_content: str, info_type: str = "kredi") -> str:
    """
    Build partial answer with disclaimer (Görsel: Fallback mekanizması için)
    
    Args:
        tr_content: Turkish answer content
        en_content: English answer content
        info_type: Type of information requested (e.g., "kredi", "credit")
    
    Returns:
        Formatted partial answer with disclaimer in both languages
    """
    disclaimer_tr = f"Bu ders için {info_type} bilgisi veri setinde doğrudan bulunamadığı için dersin genel bilgilerinden yararlanılmıştır."
    disclaimer_en = f"Since the {info_type} information for this course was not directly found in the dataset, general course information was utilized."
    
    tr = f"{tr_content}\n\nNot: {disclaimer_tr}"
    en = f"{en_content}\n\nNote: {disclaimer_en}"
    
    return format_answer(tr, en)


def build_pool_course_answer(course_code: str) -> str:
    """
    Build answer for pool courses (Görsel: Pool course logic)
    
    Args:
        course_code: Pool course code (e.g., "ELEC001", "POOL003")
    
    Returns:
        Formatted pool course explanation in both languages
    """
    tr = f"{course_code} bir havuz dersidir. Kredisi seçilen derse göre değişmektedir."
    en = f"{course_code} is a pool course. Its credit value depends on the selected elective."
    return format_answer(tr, en)
