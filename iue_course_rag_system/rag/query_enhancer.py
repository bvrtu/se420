"""
Query Enhancement Module
Improves query matching by extracting course codes and enhancing queries
"""

import re
from typing import Dict, Optional, List


class QueryEnhancer:
    """Enhances queries for better retrieval"""
    
    @staticmethod
    def extract_course_code(query: str) -> Optional[str]:
        """
        Extract course code from query (e.g., "SE 115", "FR 103", "se115")
        
        Args:
            query: User query
            
        Returns:
            Extracted course code or None
        """
        # Pattern: 2-4 letters followed by optional space and 3-4 digits
        pattern = r'\b([A-Z]{2,4})\s*(\d{3,4})\b'
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            code = match.group(1).upper()
            num = match.group(2)
            return f"{code} {num}"
        return None
    
    @staticmethod
    def enhance_query(query: str) -> str:
        """
        Enhance query with synonyms and variations
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query
        """
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
        
        # Add course code variations if found
        course_code = QueryEnhancer.extract_course_code(query)
        if course_code:
            # Add variations: "SE 115", "SE115", "se 115"
            parts = course_code.split()
            if len(parts) == 2:
                enhanced += f" {parts[0]}{parts[1]} {parts[0].lower()}{parts[1]}"
        
        return enhanced
    
    @staticmethod
    def detect_query_type(query: str) -> Dict[str, Optional[str]]:
        """
        Detect query type and extract filters
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query_type, department_filter, course_type_filter, course_code
        """
        query_lower = query.lower()
        
        # Detect department
        department_filter = None
        if 'software engineering' in query_lower or 'se' in query_lower:
            department_filter = 'Software Engineering'
        elif 'computer engineering' in query_lower or 'ce' in query_lower:
            department_filter = 'Computer Engineering'
        elif 'electrical' in query_lower or 'electronics' in query_lower or 'eee' in query_lower:
            department_filter = 'Electrical and Electronics Engineering'
        elif 'industrial engineering' in query_lower or 'ie' in query_lower:
            department_filter = 'Industrial Engineering'
        
        # Detect course type
        course_type_filter = None
        if 'mandatory' in query_lower or 'required' in query_lower or 'zorunlu' in query_lower:
            course_type_filter = 'Mandatory'
        elif 'elective' in query_lower or 'seçmeli' in query_lower:
            course_type_filter = 'Elective'
        
        # Extract course code
        course_code = QueryEnhancer.extract_course_code(query)
        
        return {
            'department_filter': department_filter,
            'course_type_filter': course_type_filter,
            'course_code': course_code
        }
