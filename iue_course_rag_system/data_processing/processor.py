"""
Data Processing Module
Cleans, normalizes, and structures scraped course data
Splits long course descriptions into semantically coherent chunks
Attaches structured metadata to each chunk
"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CourseDataProcessor:
    """Processes and structures scraped course data"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize processor
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_html_artifacts(self, text: str) -> str:
        """Remove HTML artifacts and normalize whitespace"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might be artifacts
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """Normalize and clean text"""
        if not text:
            return ""
        
        # Clean HTML artifacts
        text = self.clean_html_artifacts(text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)
        
        return text.strip()
    
    def split_into_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split long text into semantically coherent chunks
        
        Args:
            text: Text to split
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text) <= self.chunk_size:
            return [{
                'text': text,
                'chunk_index': 0,
                **metadata
            }]
        
        chunks = []
        
        # Try to split at sentence boundaries first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    **metadata
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last part of current chunk for overlap
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += (" " + sentence if current_chunk else sentence)
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': chunk_index,
                **metadata
            })
        
        return chunks
    
    def process_course(self, course: Dict) -> List[Dict]:
        """
        Process a single course into chunks with metadata
        
        Returns:
            List of course chunks with metadata
        """
        chunks = []
        
        # Extract and normalize fields
        # Görsel: Course code normalization (SE 115 / SE115) - retrieval zinciri kırılmasın
        raw_course_code = course.get('course_code', '')
        course_code = raw_course_code.replace(' ', '').upper() if raw_course_code else ''  # Görsel: Normalize
        
        course_name = self.normalize_text(course.get('course_name', ''))
        department = course.get('department', '')
        semester = course.get('semester', '')
        year = course.get('year')
        course_type = course.get('type', '')
        ects = course.get('ects')
        local_credits = course.get('local_credits')
        
        # Base metadata for all chunks
        base_metadata = {
            'department': department,
            'course_code': course_code,
            'course_name': course_name,
            'semester': semester,
            'year': year,
            'type': course_type,  # Mandatory or Elective
            'ects': ects,
            'local_credits': local_credits
        }
        
        # Create a dedicated metadata chunk with all course info (for better retrieval of course details)
        if course_code:
            metadata_text = f"Course Code: {course_code}. Course Name: {course_name}."
            if course_type:
                metadata_text += f" Type: {course_type}."
            if ects is not None:
                metadata_text += f" ECTS Credits: {ects}."
            if local_credits is not None:
                metadata_text += f" Local Credits: {local_credits}."
            if semester:
                metadata_text += f" Semester: {semester}."
            if year:
                metadata_text += f" Year: {year}."
            if department:
                metadata_text += f" Department: {department}."
            
            # Metadata chunk - credits bilgisi için önemli
            # Görsel: section değerleri standart değilse retrieval kaçar
            # Görsel: "section = section.lower().strip()" yapılmalı
            section_name = 'credits'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            
            metadata_chunk = {
                'text': metadata_text,
                'chunk_index': 0,
                **base_metadata,
                'section': section_name  # Görsel: Standardize edilmiş
            }
            chunks.append(metadata_chunk)
        
        # Process objectives
        objectives = self.normalize_text(course.get('objectives', ''))
        if objectives:
            section_name = 'objectives'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            obj_chunks = self.split_into_chunks(
                objectives,
                {**base_metadata, 'section': section_name}
            )
            chunks.extend(obj_chunks)
        
        # Process description
        description = self.normalize_text(course.get('description', ''))
        if description:
            section_name = 'description'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            desc_chunks = self.split_into_chunks(
                description,
                {**base_metadata, 'section': section_name}
            )
            chunks.extend(desc_chunks)
        
        # Process learning outcomes
        learning_outcomes = course.get('learning_outcomes', [])
        if learning_outcomes:
            if isinstance(learning_outcomes, list):
                lo_text = ' '.join([self.normalize_text(str(lo)) for lo in learning_outcomes])
            else:
                lo_text = self.normalize_text(str(learning_outcomes))
            
            if lo_text:
                section_name = 'learning_outcomes'
                section_name = section_name.lower().strip()  # Görsel: Standardize
                lo_chunks = self.split_into_chunks(
                    lo_text,
                    {**base_metadata, 'section': section_name}
                )
                chunks.extend(lo_chunks)
        
        # Process weekly topics - HER HAFTA AYRI CHUNK (ÇÖZÜM: Weekly topics ayrı chunk)
        weekly_topics = course.get('weekly_topics', [])
        if weekly_topics:
            section_name = 'weekly_topics'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            # Her hafta için ayrı chunk oluştur
            for topic in weekly_topics:
                if isinstance(topic, dict):
                    week = topic.get('week', '')
                    topic_text = topic.get('topic', '')
                    materials = topic.get('required_materials', '')
                    
                    if topic_text:
                        # Her hafta için ayrı chunk
                        week_text = f"Week {week}: {topic_text}"
                        if materials:
                            week_text += f" Required Materials: {materials}"
                        
                        week_text = self.normalize_text(week_text)
                        if week_text:
                            week_chunk = {
                                'text': week_text,
                                'chunk_index': int(week) if week.isdigit() else 0,
                                **base_metadata,
                                'section': section_name  # Görsel: Standardize edilmiş
                            }
                            chunks.append(week_chunk)
        
        # Process prerequisites
        prerequisites = self.normalize_text(course.get('prerequisites', ''))
        if prerequisites:
            section_name = 'prerequisites'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            prereq_chunks = self.split_into_chunks(
                prerequisites,
                {**base_metadata, 'section': section_name}
            )
            chunks.extend(prereq_chunks)
        
        # Process assessment information
        assessment = course.get('assessment', {})
        if assessment:
            assessment_text = ""
            
            # Semester activities
            semester_activities = assessment.get('semester_activities', [])
            if semester_activities:
                assessment_text += "Semester Activities: "
                for activity in semester_activities:
                    activity_name = activity.get('activity', '')
                    number = activity.get('number', '')
                    weighting = activity.get('weighting', '')
                    if activity_name:
                        assessment_text += f"{activity_name} ({number} activities, {weighting}% weighting). "
            
            # Weighting
            weighting = assessment.get('weighting', {})
            if weighting:
                sa = weighting.get('semester_activities', {})
                esa = weighting.get('end_of_semester_activities', {})
                total = weighting.get('total', {})
                if sa:
                    assessment_text += f"Semester Activities Weighting: {sa.get('number', '')} activities, {sa.get('percentage', '')}%. "
                if esa:
                    assessment_text += f"End-of-Semester Activities Weighting: {esa.get('number', '')} activities, {esa.get('percentage', '')}%. "
                if total:
                    assessment_text += f"Total Weighting: {total.get('number', '')} activities, {total.get('percentage', '')}%. "
            
            if assessment_text:
                assessment_text = self.normalize_text(assessment_text)
                section_name = 'assessment'
                section_name = section_name.lower().strip()  # Görsel: Standardize
                assessment_chunks = self.split_into_chunks(
                    assessment_text,
                    {**base_metadata, 'section': section_name}
                )
                chunks.extend(assessment_chunks)
        
        # Process ECTS workload
        ects_workload = course.get('ects_workload', {})
        if ects_workload:
            workload_text = ""
            activities = ects_workload.get('activities', [])
            if activities:
                workload_text += "ECTS Workload: "
                for activity in activities:
                    activity_name = activity.get('activity', '')
                    number = activity.get('number', '')
                    duration = activity.get('duration_hours', '')
                    workload = activity.get('workload', '')
                    if activity_name:
                        workload_text += f"{activity_name}: {number} x {duration}h = {workload} hours. "
            
            total_workload = ects_workload.get('total', '')
            if total_workload:
                workload_text += f"Total Workload: {total_workload} hours. "
            
            if workload_text:
                workload_text = self.normalize_text(workload_text)
                section_name = 'ects_workload'
                section_name = section_name.lower().strip()  # Görsel: Standardize
                workload_chunks = self.split_into_chunks(
                    workload_text,
                    {**base_metadata, 'section': section_name}
                )
                chunks.extend(workload_chunks)
        
        # Process available courses for SFL/ELEC/POOL
        # Create separate chunks for each nested course with full metadata
        available_courses = course.get('available_courses', [])
        if available_courses:
            # Create a summary chunk
            available_text = f"Available courses for {course_code}: "
            for nested in available_courses:
                nested_code = nested.get('course_code', '')
                nested_name = nested.get('course_name', '')
                nested_ects = nested.get('ects')
                nested_local = nested.get('local_credits')
                if nested_code:
                    available_text += f"{nested_code} - {nested_name}"
                    if nested_ects is not None:
                        available_text += f" (ECTS: {nested_ects})"
                    if nested_local is not None:
                        available_text += f" (Local Credits: {nested_local})"
                    available_text += ". "
            
            available_text = self.normalize_text(available_text)
            section_name = 'available_courses'
            section_name = section_name.lower().strip()  # Görsel: Standardize
            available_chunks = self.split_into_chunks(
                available_text,
                {**base_metadata, 'section': section_name}
            )
            chunks.extend(available_chunks)
            
            # Create individual chunks for each nested course (for better retrieval)
            for nested in available_courses:
                nested_code = nested.get('course_code', '')
                nested_name = nested.get('course_name', '')
                nested_ects = nested.get('ects')
                nested_local = nested.get('local_credits')
                nested_semester = nested.get('semester', '')
                nested_objectives = self.normalize_text(nested.get('objectives', ''))
                nested_description = self.normalize_text(nested.get('description', ''))
                
                if nested_code:
                    # Görsel: Course code normalization (SE 115 / SE115)
                    # Normalize course code: remove spaces, uppercase
                    normalized_nested_code = nested_code.replace(' ', '').upper()
                    # Store both original and normalized
                    
                    # Create metadata for nested course
                    section_name = 'nested_course'
                    section_name = section_name.lower().strip()  # Görsel: Standardize
                    nested_metadata = {
                        'department': department,  # Parent department
                        'course_code': normalized_nested_code,  # Görsel: Normalize edilmiş
                        'course_name': nested_name,
                        'semester': nested_semester,
                        'year': year,
                        'type': 'Elective',  # SFL/ELEC/POOL nested courses are typically elective
                        'ects': nested_ects,
                        'local_credits': nested_local,
                        'section': section_name,
                        'parent_course': course_code  # Track which parent course this belongs to
                    }
                    
                    # Create chunk with nested course info
                    nested_text = f"Course Code: {nested_code}. Course Name: {nested_name}."
                    if nested_ects is not None:
                        nested_text += f" ECTS Credits: {nested_ects}."
                    if nested_local is not None:
                        nested_text += f" Local Credits: {nested_local}."
                    if nested_semester:
                        nested_text += f" Semester: {nested_semester}."
                    if nested_objectives:
                        nested_text += f" Objectives: {nested_objectives}"
                    if nested_description:
                        nested_text += f" Description: {nested_description}"
                    
                    nested_text = self.normalize_text(nested_text)
                    if nested_text:
                        nested_chunk = {
                            'text': nested_text,
                            'chunk_index': 0,
                            **nested_metadata
                        }
                        chunks.append(nested_chunk)
        
        return chunks
    
    def process_all_courses(self, all_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Process all courses from all departments
        
        Args:
            all_data: Dictionary with department keys and course lists
            
        Returns:
            List of all processed chunks
        """
        all_chunks = []
        
        for dept_key, courses in all_data.items():
            logger.info(f"Processing {len(courses)} courses from {dept_key}")
            
            for course in courses:
                try:
                    chunks = self.process_course(course)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing course {course.get('course_code', 'unknown')}: {e}")
                    continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_processed_data(self, chunks: List[Dict], output_path: str):
        """Save processed chunks to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed data saved to {output_path}")
