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
