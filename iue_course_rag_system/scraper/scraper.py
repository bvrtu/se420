"""
Web Scraper for IUE Engineering Department Course Data
Scrapes curriculum and course detail pages from IUE ECTS website
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IUECourseScraper:
    """Scraper for IUE course data from ECTS website"""
    
    BASE_URL = "https://ects.ieu.edu.tr/new"
    
    @staticmethod
    def normalize_course_code(course_code: str) -> str:
        """
        Normalize course code (Görsel: SE 115 / SE115 → retrieval zinciri kırılmasın)
        
        Args:
            course_code: Raw course code (e.g., "SE 115", "SE115", "se 115")
            
        Returns:
            Normalized course code (e.g., "SE115" - uppercase, no spaces)
        """
        if not course_code:
            return ""
        # Görsel: Course code'u her zaman normalize et (SE 115 / SE115)
        normalized = course_code.replace(' ', '').upper().strip()
        return normalized
    
    # Department configurations
    DEPARTMENTS = {
        "software_engineering": {
            "name": "Software Engineering",
            "section": "se.cs.ieu.edu.tr",
            "url": "https://ects.ieu.edu.tr/new/akademik.php?sid=curr_before_2025&section=se.cs.ieu.edu.tr&lang=en",
            "elec_count": 7  # Number of ELEC courses for this department
        },
        "computer_engineering": {
            "name": "Computer Engineering",
            "section": "ce.cs.ieu.edu.tr",
            "url": "https://ects.ieu.edu.tr/new/akademik.php?section=ce.cs.ieu.edu.tr&sid=curr_before_2025&lang=en",
            "elec_count": 5  # Number of ELEC courses for this department
        },
        "electrical_electronics": {
            "name": "Electrical and Electronics Engineering",
            "section": "ete.cs.ieu.edu.tr",
            "url": "https://ects.ieu.edu.tr/new/akademik.php?section=ete.cs.ieu.edu.tr&sid=curr_before_2025&lang=en",
            "elec_count": 3  # Number of ELEC courses for this department
        },
        "industrial_engineering": {
            "name": "Industrial Engineering",
            "section": "is.cs.ieu.edu.tr",
            "url": "https://ects.ieu.edu.tr/new/akademik.php?sid=curr_before_2025&section=is.cs.ieu.edu.tr&lang=en",
            "elec_count": 6  # Number of ELEC courses for this department
        }
    }
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize the scraper
        
        Args:
            delay: Delay between requests in seconds (to be respectful)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage"""
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            time.sleep(self.delay)  # Be respectful
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def extract_sfl_courses_from_curriculum(self, sfl_link, soup: BeautifulSoup, section: str) -> List[Dict]:
        """
        Extract nested language courses for SFL from curriculum page
        First tries to fetch from SFL course detail page, then falls back to pattern matching
        """
        nested_courses = []
        
        # Language course prefixes (corrected: FR not TFR)
        lang_prefixes = ['FR', 'ITL', 'GER', 'RUS', 'SPN', 'JPN', 'CHN', 'NFR', 'CFR', 'TFR']  # Include TFR as fallback
        
        # Extract SFL course number (e.g., "SFL 201" -> "201")
        sfl_code_text = sfl_link.get_text(strip=True)
        sfl_num_match = re.search(r'SFL\s*(\d+)', sfl_code_text, re.IGNORECASE)
        sfl_number = sfl_num_match.group(1) if sfl_num_match else None
        
        # Method 1: Try to fetch from SFL course detail page
        sfl_href = sfl_link.get('href', '')
        if sfl_href:
            if not sfl_href.startswith('http'):
                sfl_href = urljoin(self.BASE_URL, sfl_href)
            
            logger.debug(f"Fetching SFL course detail page: {sfl_href}")
            detail_soup = self.fetch_page(sfl_href)
            
            if detail_soup:
                # Look for nested language courses in the detail page
                # They might be in a table, div, or list
                # Check for tables with language course links
                for table in detail_soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows:
                        lang_links = row.find_all('a', href=re.compile(r'syllabus\.php'))
                        for link in lang_links:
                            raw_code = link.get_text(strip=True)
                            code = self.normalize_course_code(raw_code)  # Görsel: Normalize
                            if any(code.startswith(prefix) for prefix in lang_prefixes):
                                name = ""
                                cells = row.find_all(['td', 'th'])
                                for i, cell in enumerate(cells):
                                    if link in cell.find_all('a'):
                                        if i + 1 < len(cells):
                                            name = cells[i + 1].get_text(strip=True)
                                        break
                                
                                href = link.get('href', '')
                                if not href.startswith('http'):
                                    href = urljoin(self.BASE_URL, href)
                                
                                if not any(c['course_code'] == code for c in nested_courses):
                                    nested_courses.append({
                                        'course_code': self.normalize_course_code(code),  # Görsel: Normalize
                                        'course_name': name,
                                        'detail_url': href
                                    })
                
                # Also check for divs or lists with language courses
                for div in detail_soup.find_all(['div', 'ul', 'ol']):
                    lang_links = div.find_all('a', href=re.compile(r'syllabus\.php'))
                    for link in lang_links:
                        raw_code = link.get_text(strip=True)
                        code = self.normalize_course_code(raw_code)  # Görsel: Normalize
                        if any(code.startswith(prefix) for prefix in lang_prefixes):
                            # Try to get name from parent element
                            name = ""
                            parent = link.find_parent(['tr', 'li', 'div', 'td'])
                            if parent:
                                # Look for text that might be the course name
                                parent_text = parent.get_text(strip=True)
                                # Remove the course code from text to get name
                                name = parent_text.replace(code, '').strip()
                            
                            href = link.get('href', '')
                            if not href.startswith('http'):
                                href = urljoin(self.BASE_URL, href)
                            
                            if not any(c['course_code'] == code for c in nested_courses):
                                nested_courses.append({
                                    'course_code': self.normalize_course_code(code),  # Görsel: Normalize
                                    'course_name': name,
                                    'detail_url': href
                                })
        
        # Method 2: If no courses found from detail page, try pattern matching in curriculum
        if not nested_courses and sfl_number:
            logger.debug(f"Method 1 found no courses, trying method 2: pattern matching in curriculum")
            all_links = soup.find_all('a', href=re.compile(r'syllabus\.php'))
            for link in all_links:
                code = link.get_text(strip=True)
                code_clean = code.replace(' ', '').upper()
                sfl_num_clean = sfl_number.strip()
                
                # Check if it's a language course matching the SFL number
                is_lang_prefix = any(code.upper().startswith(p) for p in lang_prefixes)
                # Match if course code ends with SFL number or contains it
                matches_pattern = (code_clean.endswith(sfl_num_clean) or 
                                 (sfl_num_clean in code_clean and len(sfl_num_clean) >= 2))
                
                if is_lang_prefix and matches_pattern:
                    parent_row = link.find_parent('tr')
                    name = ""
                    if parent_row:
                        cells = parent_row.find_all(['td', 'th'])
                        for i, cell in enumerate(cells):
                            if link in cell.find_all('a'):
                                if i + 1 < len(cells):
                                    name = cells[i + 1].get_text(strip=True)
                                break
                    
                    href = link.get('href', '')
                    if not href.startswith('http'):
                        href = urljoin(self.BASE_URL, href)
                    
                    if not any(c['course_code'] == code for c in nested_courses):
                        nested_courses.append({
                            'course_code': code,
                            'course_name': name,
                            'detail_url': href
                        })
        
        # Method 3: If still no courses, generate based on pattern and fetch directly
        # Based on images and user feedback: FR, ITL, GER, RUS, SPN, JPN, CHN
        # Pattern mapping for different SFL courses:
        # SFL 1013 -> FR 103, ITL 103, GER 101, RUS 101, SPN 101, JPN 101, CHN 101
        # SFL 1024 -> FR 104, ITL 104, GER 102, RUS 102, SPN 102, JPN 102, CHN 102
        # SFL 201 -> FR 201, ITL 201, GER 201, RUS 201, SPN 201, JPN 201, CHN 201
        # SFL 202 -> FR 202, ITL 202, GER 202, RUS 202, SPN 202, JPN 202, CHN 202
        if not nested_courses and sfl_number:
            logger.debug(f"Method 2 found no courses, trying method 3: generating from pattern and fetching directly")
            # Standard language codes (corrected: FR not TFR)
            standard_langs = ['FR', 'ITL', 'GER', 'RUS', 'SPN', 'JPN', 'CHN']
            
            # Pattern mapping for SFL course numbers to language course numbers
            sfl_patterns = {
                '1013': {'FR': '103', 'ITL': '103', 'GER': '101', 'RUS': '101', 'SPN': '101', 'JPN': '101', 'CHN': '101'},
                '1024': {'FR': '104', 'ITL': '104', 'GER': '102', 'RUS': '102', 'SPN': '102', 'JPN': '102', 'CHN': '102'},
                '201': {'FR': '201', 'ITL': '201', 'GER': '201', 'RUS': '201', 'SPN': '201', 'JPN': '201', 'CHN': '201'},
                '202': {'FR': '202', 'ITL': '202', 'GER': '202', 'RUS': '202', 'SPN': '202', 'JPN': '202', 'CHN': '202'}
            }
            
            # Get the pattern for this SFL course
            lang_course_numbers = sfl_patterns.get(sfl_number)
            if not lang_course_numbers:
                # Fallback: use SFL number directly for all languages
                lang_course_numbers = {lang: sfl_number for lang in standard_langs}
            
            for lang_code in standard_langs:
                # Get the language course number from pattern
                lang_num = lang_course_numbers.get(lang_code, sfl_number)
                # Construct course code: e.g., "FR 201" or "FR 103" for SFL 1013
                raw_lang_course_code = f"{lang_code} {lang_num}"
                lang_course_code = self.normalize_course_code(raw_lang_course_code)  # Görsel: Normalize
                
                # Try to find this course in curriculum first
                found_in_curriculum = False
                for link in soup.find_all('a', href=re.compile(r'syllabus\.php')):
                    raw_code = link.get_text(strip=True)
                    code = self.normalize_course_code(raw_code)  # Görsel: Normalize
                    # Match with flexible spacing
                    if code == lang_course_code:  # Görsel: Normalize edilmiş, direkt karşılaştır
                        parent_row = link.find_parent('tr')
                        name = ""
                        if parent_row:
                            cells = parent_row.find_all(['td', 'th'])
                            for i, cell in enumerate(cells):
                                if link in cell.find_all('a'):
                                    if i + 1 < len(cells):
                                        name = cells[i + 1].get_text(strip=True)
                                    break
                        
                        href = link.get('href', '')
                        if not href.startswith('http'):
                            href = urljoin(self.BASE_URL, href)
                        
                        nested_courses.append({
                            'course_code': code,  # Görsel: Normalize edilmiş
                            'course_name': name,
                            'detail_url': href
                        })
                        found_in_curriculum = True
                        break
                
                # If not found in curriculum, construct URL and fetch directly
                if not found_in_curriculum:
                    encoded_code = lang_course_code.replace(" ", "%20")
                    # Use cer=0&sem=1 format (without currType) for proper page loading
                    lang_course_url = f"{self.BASE_URL}/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1"
                    
                    # Fetch the language course detail page to get full information
                    logger.debug(f"Fetching language course directly: {lang_course_url}")
                    lang_detail = self.scrape_course_detail(lang_course_url, lang_course_code)
                    
                    if lang_detail and lang_detail.get('course_name'):
                        # Use the full scraped detail information
                        nested_courses.append({
                            'course_code': lang_course_code,  # Görsel: Zaten normalize edilmiş
                            'course_name': lang_detail.get('course_name', ''),
                            'detail_url': lang_course_url,
                            'objectives': lang_detail.get('objectives', ''),
                            'description': lang_detail.get('description', ''),
                            'weekly_topics': lang_detail.get('weekly_topics', []),
                            'learning_outcomes': lang_detail.get('learning_outcomes', []),
                            'assessment': lang_detail.get('assessment', {}),
                            'ects_workload': lang_detail.get('ects_workload', {}),
                            'theory_hours': lang_detail.get('theory_hours'),
                            'application_hours': lang_detail.get('application_hours'),
                            'local_credits': lang_detail.get('local_credits'),
                            'ects': lang_detail.get('ects')
                        })
                        logger.debug(f"Found language course {lang_course_code} via direct fetch with full details")
                    else:
                        logger.debug(f"Language course {lang_course_code} not found (404 or error)")
        
        logger.debug(f"Found {len(nested_courses)} nested courses for {sfl_code_text}")
        return nested_courses
    
    def extract_sfl_courses(self, sfl_code: str, section: str) -> List[Dict]:
        """
        Extract nested courses for SFL (Second Foreign Languages) courses
        This is a fallback - should use extract_sfl_courses_from_curriculum instead
        """
        nested_courses = []
        logger.warning(f"Using fallback method for {sfl_code} - nested courses should be extracted from curriculum page")
        return nested_courses
    
    def extract_nested_courses_from_link(self, link, section: str) -> List[Dict]:
        """
        Extract nested courses from SFL/ELEC/POOL links
        Checks for pool links (akademik.php?sid=pool) and extracts from those pages
        """
        nested_courses = []
        raw_course_code = link.get_text(strip=True)
        course_code = self.normalize_course_code(raw_course_code)  # Görsel: Normalize
        
        # Check if this is a POOL link (not a syllabus link)
        href = link.get('href', '')
        if 'sid=pool' in href or 'pool' in href.lower():
            # This is a POOL link - extract from pool page
            if not href.startswith('http'):
                href = urljoin(self.BASE_URL, href)
            
            pool_soup = self.fetch_page(href)
            if pool_soup:
                # Extract courses from pool table
                tables = pool_soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:  # Skip header
                        cells = row.find_all(['td', 'th'])
                        code_link = row.find('a', href=re.compile(r'syllabus\.php'))
                        if code_link:
                            code = code_link.get_text(strip=True)
                            name = ""
                            for i, cell in enumerate(cells):
                                if code_link in cell.find_all('a'):
                                    if i + 1 < len(cells):
                                        name = cells[i + 1].get_text(strip=True)
                                    break
                            
                            nested_href = code_link.get('href', '')
                            if not nested_href.startswith('http'):
                                nested_href = urljoin(self.BASE_URL, nested_href)
                            
                            # Extract ECTS
                            ects = None
                            for cell in cells:
                                text = cell.get_text(strip=True)
                                if text.isdigit():
                                    header = cell.find_previous(['th'])
                                    if header and 'ects' in header.get_text(strip=True).lower():
                                        try:
                                            ects = int(text)
                                        except:
                                            pass
                            
                            nested_courses.append({
                                'course_code': self.normalize_course_code(code),  # Görsel: Normalize
                                'course_name': name,
                                'detail_url': nested_href,
                                'ects': ects
                            })
        
        # For SFL courses, use special extraction
        elif course_code.startswith('SFL'):
            nested_courses = self.extract_sfl_courses(course_code, section)
        
        return nested_courses
    
    def _build_course_url(self, course_code: str, section: str) -> str:
        """Build course detail URL from course code and section"""
        encoded_code = course_code.replace(" ", "%20")
        # Use cer=0&sem=1 format (without currType) for proper page loading with correct content
        return f"{self.BASE_URL}/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1"
    
    def extract_course_links_from_curriculum(self, soup: BeautifulSoup, section: str, scraped_course_codes: set = None) -> List[Dict]:
        """
        Extract course links from the curriculum page
        Handles regular courses, SFL, ELEC, and POOL courses
        
        Returns list of course info dicts with code, name, and detail URL
        """
        courses = []
        seen_course_codes = set()  # Track seen course codes to avoid duplicates
        
        # Find all course code links in tables
        # Course codes are typically in <a> tags with href containing 'syllabus.php'
        course_links = soup.find_all('a', href=re.compile(r'syllabus\.php'))
        
        for link in course_links:
            try:
                raw_course_code = link.text.strip()
                course_code = self.normalize_course_code(raw_course_code)  # Görsel: Normalize
                
                # Skip if already seen (for non-SFL/ELEC/POOL courses)
                if not (course_code.startswith('SFL') or course_code.startswith('ELEC') or course_code.startswith('POOL')):
                    if course_code in seen_course_codes:
                        continue  # Skip duplicate
                    seen_course_codes.add(course_code)
                
                course_name = ""
                
                # Find the course name - usually in the next cell or same row
                parent_row = link.find_parent('tr')
                if parent_row:
                    cells = parent_row.find_all(['td', 'th'])
                    for i, cell in enumerate(cells):
                        if link in cell.find_all('a'):
                            # Course name is usually in the next cell
                            if i + 1 < len(cells):
                                course_name = cells[i + 1].get_text(strip=True)
                            break
                
                # Get the full URL
                href = link.get('href', '')
                if href:
                    if href.startswith('http'):
                        course_url = href
                    else:
                        course_url = urljoin(self.BASE_URL, href)
                    
                    # Extract semester info from table structure
                    semester_info = self._extract_semester_info(parent_row)
                    
                    course_data = {
                        'course_code': self.normalize_course_code(course_code),  # Görsel: Normalize
                        'course_name': course_name,
                        'detail_url': course_url,
                        'semester': semester_info.get('semester'),
                        'year': semester_info.get('year'),
                        'type': semester_info.get('type'),  # Mandatory/Elective
                        'ects': semester_info.get('ects'),
                        'local_credits': semester_info.get('local_credits'),
                        'available_courses': []  # For SFL/ELEC/POOL nested courses
                    }
                    
                    # Check if this is SFL course
                    if course_code.startswith('SFL'):
                        # SFL courses are the same across all departments
                        # Check if this SFL course was already scraped in another department
                        if scraped_course_codes and course_code in scraped_course_codes:
                            continue  # Skip this SFL course entirely
                        
                        # SFL courses have nested language courses - extract from curriculum
                        nested = self.extract_sfl_courses_from_curriculum(link, soup, section)
                        if nested:
                            # Filter out duplicate nested courses (cross-department check)
                            filtered_nested = []
                            for n in nested:
                                nested_code = n.get('course_code', '')
                                if nested_code:
                                    # Check if already scraped (cross-department)
                                    if scraped_course_codes and nested_code in scraped_course_codes:
                                        continue
                                    # Check local seen_course_codes
                                    if nested_code not in seen_course_codes:
                                        seen_course_codes.add(nested_code)
                                        if scraped_course_codes is not None:
                                            scraped_course_codes.add(nested_code)
                                        filtered_nested.append(n)
                            if filtered_nested:
                                course_data['available_courses'] = filtered_nested
                                lang_codes = ', '.join([c['course_code'] for c in filtered_nested])
                            course_data['description'] = f"Second Foreign Languages course. Available language options: {lang_codes}"
                    elif course_code.startswith('POOL'):
                        # POOL courses are handled separately in _extract_pool_courses_from_links
                        # Skip adding here, will be added later
                        continue
                    
                    courses.append(course_data)
            except Exception as e:
                logger.warning(f"Error extracting course link: {e}")
                continue
        
        # Extract ELEC courses from "Elective Courses" section
        elective_courses = self._extract_elective_courses_table(soup, section)
        
        # Filter out duplicate elective courses (check both seen_course_codes and scraped_course_codes)
        filtered_elective_courses = []
        for elec in elective_courses:
            elec_code = elec.get('course_code', '')
            if elec_code:
                # Check both sets
                if elec_code not in seen_course_codes:
                    seen_course_codes.add(elec_code)
                    # Also check scraped_course_codes if provided
                    if scraped_course_codes is None or elec_code not in scraped_course_codes:
                        if scraped_course_codes is not None:
                            scraped_course_codes.add(elec_code)
                        filtered_elective_courses.append(elec)
        elective_courses = filtered_elective_courses
        
        # Get department ELEC count
        dept_key = None
        elec_count = None
        for key, dept_info in self.DEPARTMENTS.items():
            if dept_info['section'] == section:
                dept_key = key
                elec_count = dept_info.get('elec_count', 0)
                break
        
        # Find ELEC parent courses (ELEC 001, ELEC 002, etc.) from curriculum
        elec_parent_courses = {}
        elec_links = soup.find_all('a', href=re.compile(r'syllabus\.php'))
        for link in elec_links:
            raw_code = link.get_text(strip=True)
            code = self.normalize_course_code(raw_code)  # Görsel: Normalize
            if code.startswith('ELEC'):
                # This is an ELEC parent course
                parent_row = link.find_parent('tr')
                semester_info = self._extract_semester_info(parent_row) if parent_row else {}
                
                elec_parent_courses[code] = {
                    'course_code': code,  # Görsel: Normalize edilmiş
                    'course_name': 'Elective Course',
                    'detail_url': '',
                    'semester': semester_info.get('semester'),
                    'year': semester_info.get('year'),
                    'type': 'Elective',
                    'ects': semester_info.get('ects'),
                    'local_credits': semester_info.get('local_credits'),
                    'available_courses': []
                }
        
        # If no ELEC parents found in curriculum, create them based on elec_count
        if not elec_parent_courses and elec_count:
            for i in range(1, elec_count + 1):
                raw_elec_code = f"ELEC {str(i).zfill(3)}"
                elec_code = self.normalize_course_code(raw_elec_code)  # Görsel: Normalize
                elec_parent_courses[elec_code] = {
                    'course_code': elec_code,  # Görsel: Normalize edilmiş
                    'course_name': 'Elective Course',
                    'detail_url': '',
                    'semester': None,  # Will be determined from nested courses
                    'year': None,
                    'type': 'Elective',
                    'ects': None,
                    'local_credits': None,
                    'available_courses': []
                }
        
        # Fetch semester info for each elective course and assign to matching ELEC parents
        # First, fetch semester info for all elective courses
        for elective in elective_courses:
            elective['semester'] = None  # Will be fetched from detail page
            
            # Fetch course detail to get semester (skip if already scraped)
            elective_code = elective.get('course_code', '')
            if scraped_course_codes and elective_code in scraped_course_codes:
                continue  # Already scraped, skip
            
            if elective.get('detail_url'):
                detail = self.scrape_course_detail(elective['detail_url'], elective_code, scraped_course_codes)
                if detail:
                    # Mark as scraped
                    if scraped_course_codes is not None:
                        scraped_course_codes.add(elective_code)
                    elective['semester'] = detail.get('semester', '')
                    # Also update other fields if available
                    if detail.get('theory_hours'):
                        elective['theory_hours'] = detail.get('theory_hours')
                    if detail.get('application_hours'):
                        elective['application_hours'] = detail.get('application_hours')
                    if detail.get('ects'):
                        elective['ects'] = detail.get('ects')
                    if detail.get('local_credits'):
                        elective['local_credits'] = detail.get('local_credits')
        
        # Assign elective courses to ELEC parents based on semester match
        # If ELEC parent has a semester, only assign matching electives
        # If ELEC parent has no semester, assign all electives
        for elec_code, elec_parent in elec_parent_courses.items():
            elec_semester = elec_parent.get('semester')
            
            for elective in elective_courses:
                elective_semester = elective.get('semester', '')
                
                # Match if:
                # 1. ELEC has no semester specified, OR
                # 2. Semesters match (exact match or both contain Fall/Spring)
                should_assign = False
                if not elec_semester:
                    should_assign = True
                elif elective_semester:
                    # Normalize semesters for comparison
                    elec_sem_lower = elec_semester.lower()
                    elective_sem_lower = elective_semester.lower()
                    
                    if elec_sem_lower == elective_sem_lower:
                        should_assign = True
                    elif 'fall' in elec_sem_lower and 'fall' in elective_sem_lower:
                        should_assign = True
                    elif 'spring' in elec_sem_lower and 'spring' in elective_sem_lower:
                        should_assign = True
                    elif 'fall/spring' in elec_sem_lower or 'fall/spring' in elective_sem_lower:
                        should_assign = True
                
                if should_assign:
                    # Fetch full course detail (like SFL/POOL)
                    elective_code = elective.get('course_code', '')
                    # Skip if already scraped
                    if scraped_course_codes and elective_code in scraped_course_codes:
                        continue
                    
                    if elective.get('detail_url'):
                        course_detail = self.scrape_course_detail(elective['detail_url'], elective_code, scraped_course_codes)
                        if course_detail:
                            # Mark as scraped
                            if scraped_course_codes is not None:
                                scraped_course_codes.add(elective_code)
                            if course_detail.get('course_name'):
                                elective.update({
                                    'objectives': course_detail.get('objectives', ''),
                                    'description': course_detail.get('description', ''),
                                    'weekly_topics': course_detail.get('weekly_topics', []),
                                    'learning_outcomes': course_detail.get('learning_outcomes', []),
                                    'assessment': course_detail.get('assessment', {}),
                                    'ects_workload': course_detail.get('ects_workload', {}),
                                    'prerequisites': course_detail.get('prerequisites', '')
                                })
                    
                    # Check if this elective course is already added (avoid duplicates)
                    elec_code = elective.get('course_code', '')
                    if elec_code and elec_code not in seen_course_codes:
                        seen_course_codes.add(elec_code)
                    elec_parent['available_courses'].append(elective)
            
            # Set description
            elec_parent['description'] = f"This elective course group contains {len(elec_parent['available_courses'])} available courses."
            if elec_parent.get('semester'):
                elec_parent['description'] += f" Semester: {elec_parent['semester']}."
        
        # Add ELEC parent courses
        for elec_parent in elec_parent_courses.values():
            courses.append(elec_parent)
        
        # Extract POOL courses from pool links (akademik.php?sid=pool)
        pool_courses = self._extract_pool_courses_from_links(soup, section, scraped_course_codes)
        courses.extend(pool_courses)
        
        # Remove duplicates based on course_code
        seen = set()
        unique_courses = []
        for course in courses:
            if course['course_code'] and course['course_code'] not in seen:
                seen.add(course['course_code'])
                unique_courses.append(course)
        
        logger.info(f"Found {len(unique_courses)} unique courses")
        return unique_courses
    
    def _extract_elective_courses_table(self, soup: BeautifulSoup, section: str) -> List[Dict]:
        """Extract elective courses from the 'Elective Courses' table"""
        elective_courses = []
        target_table = None
        
        # Method 1: Find table with class="elective" (most reliable)
        tables = soup.find_all('table', class_=re.compile(r'elective', re.I))
        if tables:
            target_table = tables[0]
            logger.debug("Found elective courses table by class='elective'")
        
        # Method 2: Find table with "Elective Courses" in title row
        if not target_table:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 0:
                    # Check first row for "Elective Courses" title
                    first_row = rows[0]
                    first_row_text = first_row.get_text(strip=True).lower()
                    if 'elective course' in first_row_text and 'explanation' not in first_row_text:
                        # Check if it has the right structure (Code, Course Name columns)
                        if len(rows) > 1:
                            header_row = rows[1] if 'code' in rows[1].get_text(strip=True).lower() else rows[0]
                            header_text = header_row.get_text(strip=True).lower()
                            if 'code' in header_text and 'course' in header_text:
                                target_table = table
                                logger.debug("Found elective courses table by title row")
                                break
        
        # Method 3: Find table with "Code" and "Course Name" columns that has syllabus links
        if not target_table:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                if len(rows) > 0:
                    header_row = rows[0]
                    header_cells = header_row.find_all(['th', 'td'])
                    header_text = ' '.join([cell.get_text(strip=True).lower() for cell in header_cells])
                    # Check if this looks like an elective courses table
                    if 'code' in header_text and 'course' in header_text and 'name' in header_text:
                        # Check if there are syllabus links (elective courses have detail pages)
                        has_syllabus_links = False
                        for row in rows[1:min(6, len(rows))]:  # Check first 5 data rows
                            if row.find('a', href=re.compile(r'syllabus\.php')):
                                has_syllabus_links = True
                                break
                        if has_syllabus_links:
                            target_table = table
                            logger.debug("Found elective courses table by structure")
                            break
        
        if target_table:
            rows = target_table.find_all('tr')
            if len(rows) >= 2:
                # Find header row (might be first or second row)
                # First row might be title row, second row is usually header
                header_row = None
                header_row_idx = 0
                
                for i, row in enumerate(rows):
                    cells = row.find_all(['th', 'td'])
                    if len(cells) > 0:
                        row_text = ' '.join([cell.get_text(strip=True).lower() for cell in cells])
                        # Check if this looks like a header row (has "code" and "course")
                        if 'code' in row_text and 'course' in row_text and 'name' in row_text:
                            header_row = row
                            header_row_idx = i
                            break
                
                if not header_row:
                    # Fallback: use first row as header
                    header_row = rows[0]
                    header_row_idx = 0
                
                # Map column indices from header row
                header_cells = header_row.find_all(['th', 'td'])
                col_indices = {}
                
                for i, cell in enumerate(header_cells):
                    header = cell.get_text(strip=True).lower()
                    if 'code' in header:
                        col_indices['code'] = i
                    elif 'course' in header and 'name' in header:
                        col_indices['name'] = i
                    elif 'theory' in header:
                        col_indices['theory'] = i
                    elif ('app' in header and 'lab' in header) or 'application' in header or 'lab' in header:
                        col_indices['app_lab'] = i
                    elif 'local' in header and 'credit' in header:
                        col_indices['local_credits'] = i
                    elif 'ects' in header:
                        col_indices['ects'] = i
                
                # Process data rows (skip title and header rows)
                for row in rows[header_row_idx + 1:]:  # Skip title and header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) < 3:
                        continue
                    
                    # Find course code link
                    code_link = row.find('a', href=re.compile(r'syllabus\.php'))
                    if not code_link:
                        continue
                    
                    code = code_link.get_text(strip=True)
                    if not code:
                        continue
                    
                    # Get course name
                    name = ""
                    if 'name' in col_indices and col_indices['name'] < len(cells):
                        name = cells[col_indices['name']].get_text(strip=True)
                    
                    # Get detail URL
                    href = code_link.get('href', '')
                    if not href.startswith('http'):
                        href = urljoin(self.BASE_URL, href)
                    
                    # Extract values from cells based on column indices
                    ects = None
                    local_credits = None
                    theory_hours = None
                    app_hours = None
                    
                    if 'ects' in col_indices and col_indices['ects'] < len(cells):
                        ects_text = cells[col_indices['ects']].get_text(strip=True)
                        if ects_text.isdigit():
                            ects = int(ects_text)
                    
                    if 'local_credits' in col_indices and col_indices['local_credits'] < len(cells):
                        credits_text = cells[col_indices['local_credits']].get_text(strip=True)
                        if credits_text.isdigit():
                            local_credits = int(credits_text)
                    
                    if 'theory' in col_indices and col_indices['theory'] < len(cells):
                        theory_text = cells[col_indices['theory']].get_text(strip=True)
                        if theory_text.isdigit():
                            theory_hours = int(theory_text)
                    
                    if 'app_lab' in col_indices and col_indices['app_lab'] < len(cells):
                        app_text = cells[col_indices['app_lab']].get_text(strip=True)
                        if app_text.isdigit():
                            app_hours = int(app_text)
                    
                    # Görsel: Course code normalization
                    normalized_code = self.normalize_course_code(code)
                    
                    elective_courses.append({
                        'course_code': normalized_code,  # Görsel: Normalize edilmiş
                        'course_name': name,
                        'detail_url': href,
                        'type': 'Elective',
                        'ects': ects,
                        'local_credits': local_credits,
                        'theory_hours': theory_hours,
                        'application_hours': app_hours
                    })
        
        return elective_courses
    
    def _extract_pool_courses_from_links(self, soup: BeautifulSoup, section: str, scraped_course_codes: set = None) -> List[Dict]:
        """
        Extract POOL courses by finding pool links (akademik.php?sid=pool) 
        and scraping those pages
        """
        pool_courses = []
        
        # Find all links that might be POOL links
        # POOL links can be: akademik.php?sid=pool or links with POOL in text
        all_links = soup.find_all('a', href=True)
        
        for link in all_links:
            href = link.get('href', '')
            link_text = link.get_text(strip=True)
            
            # Check if this is a POOL link
            is_pool_link = False
            pool_code = None
            
            # Method 1: Check if href contains pool
            if 'sid=pool' in href or ('pool' in href.lower() and 'akademik.php' in href):
                is_pool_link = True
                # Try to extract pool code from surrounding context
                parent = link.find_parent(['tr', 'td', 'div', 'h1', 'h2', 'h3', 'h4'])
                if parent:
                    parent_text = parent.get_text()
                    pool_match = re.search(r'POOL\s*(\d+)', parent_text, re.IGNORECASE)
                    if pool_match:
                        pool_code = f"POOL {pool_match.group(1).zfill(3)}"
            
            # Method 2: Check if link text contains POOL
            if not is_pool_link:
                pool_match = re.search(r'POOL\s*(\d+)', link_text, re.IGNORECASE)
                if pool_match:
                    is_pool_link = True
                    pool_code = f"POOL {pool_match.group(1).zfill(3)}"
                    # Check if href is a pool page
                    if 'akademik.php' in href:
                        if not href.startswith('http'):
                            href = urljoin(self.BASE_URL, href)
                    else:
                        # Construct pool URL
                        href = f"{self.BASE_URL}/akademik.php?sid=pool&section={section}&currType=before_2025"
            
            if is_pool_link and pool_code:
                if not href.startswith('http'):
                    href = urljoin(self.BASE_URL, href)
                
                # Ensure section is in URL
                if 'section=' not in href:
                    href += f"&section={section}" if '?' in href else f"?section={section}"
                
                # Get semester from parent row
                pool_semester = None
                parent_row = link.find_parent('tr')
                if parent_row:
                    semester_info = self._extract_semester_info(parent_row)
                    pool_semester = semester_info.get('semester')
                
                # Fetch the pool page
                pool_soup = self.fetch_page(href)
                if pool_soup:
                    pool_data = self._extract_pool_courses(pool_soup, section, pool_code, pool_semester, scraped_course_codes)
                    if pool_data:
                        # Check if we already have this pool
                        if not any(p['course_code'] == pool_code for p in pool_courses):
                            pool_courses.append(pool_data)
        
        return pool_courses
    
    def _extract_pool_courses(self, soup: BeautifulSoup, section: str, pool_code: str = None, pool_semester: str = None, scraped_course_codes: set = None) -> Dict:
        """
        Extract POOL courses (POOL 003, POOL 004, POOL 006)
        Returns dict with pool info and nested courses, including minimum ECTS requirements
        
        Args:
            soup: BeautifulSoup object of the pool page
            section: Department section
            pool_code: POOL course code (e.g., "POOL 003")
            pool_semester: Semester when this POOL is taken (e.g., "Fall", "Spring") - used for filtering nested courses
        """
        # Minimum ECTS requirements per department and pool
        # SE: POOL 003 (min 4 ECTS), POOL 004 (min 5 ECTS), POOL 006 (min 4 ECTS)
        # CE: POOL 003 (min 4 ECTS), POOL 005 (min 5 ECTS), POOL 006 (min 5 ECTS)
        # IS: POOL 004 (min 4 ECTS), POOL 006 (min 5 ECTS)
        # ETE: POOL 003 (min 4 ECTS), POOL 004 (min 4 ECTS), POOL 006 (min 4 ECTS)
        min_ects_requirements = {
            "software_engineering": {
                "POOL 003": 4,
                "POOL 004": 5,
                "POOL 006": 4
            },
            "computer_engineering": {
                "POOL 003": 4,
                "POOL 005": 5,
                "POOL 006": 5
            },
            "electrical_electronics": {
                "POOL 003": 4,
                "POOL 004": 4,
                "POOL 006": 4
            },
            "industrial_engineering": {
                "POOL 004": 4,
                "POOL 006": 5
            }
        }
        
        # Find department key from section
        dept_key = None
        for key, dept_info in self.DEPARTMENTS.items():
            if dept_info['section'] == section:
                dept_key = key
                break
        
        # If pool_code not provided, try to find it from headings
        if not pool_code:
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
            for heading in headings:
                heading_text = heading.get_text(strip=True)
                if 'POOL' in heading_text.upper():
                    pool_match = re.search(r'POOL\s*(\d+)', heading_text, re.IGNORECASE)
                    if pool_match:
                        pool_code = f"POOL {pool_match.group(1).zfill(3)}"
                        break
        
        if not pool_code:
            return None
        
        # Find pool name from heading
        pool_name = pool_code
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            if pool_code.replace(' ', '') in heading_text.replace(' ', '').upper() or 'POOL' in heading_text.upper():
                pool_name = heading_text.strip()
                # Extract full pool name (e.g., "POOL 003 - GEC-Social Sciences A: Economics")
                break
        
        # Find the table with pool courses
        # Note: Multiple POOLs can be in the same table, need to find the right section
        nested_courses = []
        tables = soup.find_all('table')
        
        # Find which rows belong to this pool_code
        target_pool_found = False
        current_pool = None
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 2:  # Need at least header + 1 row
                continue
            
            # Map column indices from first header row
            header_row = None
            col_indices = {}
            
            # Find header row (usually first row with "Code" in it)
            for row in rows:
                cells = row.find_all(['th', 'td'])
                if len(cells) >= 3:
                    header_text = ' '.join([cell.get_text(strip=True).lower() for cell in cells])
                    if 'code' in header_text and ('course' in header_text or 'name' in header_text):
                        header_row = row
                        header_cells = cells
                        # Map column indices
                        for i, cell in enumerate(header_cells):
                            header = cell.get_text(strip=True).lower()
                            if 'code' in header:
                                col_indices['code'] = i
                            elif 'course' in header and 'name' in header:
                                col_indices['name'] = i
                            elif 'theory' in header:
                                col_indices['theory'] = i
                            elif ('app' in header and 'lab' in header) or 'application' in header or 'lab' in header:
                                col_indices['app_lab'] = i
                            elif 'local' in header and 'credit' in header:
                                col_indices['local_credits'] = i
                            elif 'ects' in header:
                                col_indices['ects'] = i
                        break
            
            if not header_row:
                continue
            
            # Process rows - look for POOL headings and course rows
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) == 0:
                    continue
                
                # Check if this is a POOL heading row (usually has POOL in first cell)
                first_cell_text = cells[0].get_text(strip=True).upper()
                if 'POOL' in first_cell_text:
                    # Extract POOL number
                    pool_match = re.search(r'POOL\s*(\d+)', first_cell_text, re.IGNORECASE)
                    if pool_match:
                        found_pool_code = f"POOL {pool_match.group(1).zfill(3)}"
                        if found_pool_code == pool_code:
                            current_pool = found_pool_code
                            target_pool_found = True
                            logger.debug(f"Found target pool section: {pool_code}")
                        else:
                            # Different pool, stop collecting courses
                            if target_pool_found:
                                break  # We've passed our pool section
                            current_pool = None
                    continue
                
                # If we're in the right pool section, process course rows
                if target_pool_found and current_pool == pool_code:
                    # Check if this is a course row (has syllabus link)
                    code_link = row.find('a', href=re.compile(r'syllabus\.php'))
                    if not code_link:
                        continue
                    
                    code = code_link.get_text(strip=True)
                    if not code:
                        continue
                    
                    # Get course name
                    name = ""
                    if 'name' in col_indices and col_indices['name'] < len(cells):
                        name = cells[col_indices['name']].get_text(strip=True)
                    
                    # Get detail URL
                    href = code_link.get('href', '')
                    if not href.startswith('http'):
                        href = urljoin(self.BASE_URL, href)
                    
                    # Extract values from cells based on column indices
                    ects = None
                    local_credits = None
                    theory_hours = None
                    app_hours = None
                    
                    if 'ects' in col_indices and col_indices['ects'] < len(cells):
                        ects_text = cells[col_indices['ects']].get_text(strip=True)
                        if ects_text.isdigit():
                            ects = int(ects_text)
                    
                    if 'local_credits' in col_indices and col_indices['local_credits'] < len(cells):
                        credits_text = cells[col_indices['local_credits']].get_text(strip=True)
                        if credits_text.isdigit():
                            local_credits = int(credits_text)
                    
                    if 'theory' in col_indices and col_indices['theory'] < len(cells):
                        theory_text = cells[col_indices['theory']].get_text(strip=True)
                        if theory_text.isdigit():
                            theory_hours = int(theory_text)
                    
                    if 'app_lab' in col_indices and col_indices['app_lab'] < len(cells):
                        app_text = cells[col_indices['app_lab']].get_text(strip=True)
                        if app_text.isdigit():
                            app_hours = int(app_text)
                    
                    # Fetch full course detail (like SFL)
                    # Check if already scraped (cross-department check)
                    if scraped_course_codes and code in scraped_course_codes:
                        logger.debug(f"Skipping duplicate POOL nested course (already scraped): {code}")
                        continue
                    
                    course_detail = None
                    if href:
                        course_detail = self.scrape_course_detail(href, code, scraped_course_codes)
                        # Mark as scraped
                        if scraped_course_codes is not None:
                            scraped_course_codes.add(code)
                    
                    # Check semester match if pool_semester is specified
                    if pool_semester and course_detail:
                        course_semester = course_detail.get('semester', '')
                        if course_semester:
                            # Normalize semesters for comparison
                            pool_sem_lower = pool_semester.lower()
                            course_sem_lower = course_semester.lower()
                            
                            # Check if semesters match
                            semester_match = False
                            if pool_sem_lower == course_sem_lower:
                                semester_match = True
                            elif 'fall' in pool_sem_lower and 'fall' in course_sem_lower:
                                semester_match = True
                            elif 'spring' in pool_sem_lower and 'spring' in course_sem_lower:
                                semester_match = True
                            elif 'fall/spring' in pool_sem_lower or 'fall/spring' in course_sem_lower:
                                semester_match = True
                            
                            # If semester doesn't match, skip this course
                            if not semester_match:
                                logger.debug(f"Skipping {code} - semester mismatch: pool={pool_semester}, course={course_semester}")
                                continue
                    
                    # Görsel: Course code normalization
                    normalized_code = self.normalize_course_code(code)
                    
                    nested_course = {
                        'course_code': normalized_code,  # Görsel: Normalize edilmiş
                        'course_name': name,
                        'detail_url': href,
                        'ects': ects,
                        'local_credits': local_credits,
                        'theory_hours': theory_hours,
                        'application_hours': app_hours
                    }
                    
                    # Add full detail if available
                    if course_detail and course_detail.get('course_name'):
                        nested_course.update({
                            'objectives': course_detail.get('objectives', ''),
                            'description': course_detail.get('description', ''),
                            'weekly_topics': course_detail.get('weekly_topics', []),
                            'learning_outcomes': course_detail.get('learning_outcomes', []),
                            'assessment': course_detail.get('assessment', {}),
                            'ects_workload': course_detail.get('ects_workload', {}),
                            'prerequisites': course_detail.get('prerequisites', ''),
                            'semester': course_detail.get('semester', '')
                        })
                    
                    # Check for duplicates before adding (normalized code ile)
                    if not any(c.get('course_code') == normalized_code for c in nested_courses):
                        nested_courses.append(nested_course)
        
        # Get minimum ECTS requirement
        min_ects = None
        if dept_key and dept_key in min_ects_requirements:
            min_ects = min_ects_requirements[dept_key].get(pool_code)
        
        # Görsel: Course code normalization
        normalized_pool_code = self.normalize_course_code(pool_code) if pool_code else ''
        
        return {
            'course_code': normalized_pool_code,  # Görsel: Normalize edilmiş
            'course_name': pool_name,
            'detail_url': '',  # POOL courses don't have detail pages
            'type': 'Pool',
            'available_courses': nested_courses,
            'minimum_ects': min_ects,
            'description': f"This pool contains {len(nested_courses)} courses. " + 
                          (f"Minimum ECTS requirement: {min_ects}" if min_ects else "No minimum ECTS requirement.")
        }
    
    def _extract_semester_info(self, row) -> Dict:
        """Extract semester, year, type, and credits from table row"""
        info = {
            'semester': None,
            'year': None,
            'type': None,
            'ects': None,
            'local_credits': None
        }
        
        if not row:
            return info
        
        # Try to find semester/year from table headers or previous rows
        # Look for text like "1. Year Fall Semester" or "2. Year Spring Semester"
        parent_table = row.find_parent('table')
        text = ""
        if parent_table:
            # Method 1: Check first row of table (title row with class="title")
            first_row = parent_table.find('tr')
            if first_row:
                first_row_cells = first_row.find_all(['td', 'th'])
                if first_row_cells:
                    first_cell = first_row_cells[0]
                    # Check if it's a title row (has class="title" or colspan)
                    if first_cell.get('class') and 'title' in first_cell.get('class'):
                        text = first_cell.get_text()
                    # Or check if first row has colspan (title row)
                    elif first_cell.get('colspan'):
                        text = first_cell.get_text()
            
            # Method 2: Check previous siblings for header (fallback)
            if not text:
                prev = row.find_previous_sibling(['tr', 'h2', 'h3', 'h4'])
                if prev:
                    text = prev.get_text()
            
            if text:
                # Extract year (1-4) and semester (Fall/Spring)
                year_match = re.search(r'(\d+)\.?\s*Year', text, re.IGNORECASE)
                if year_match:
                    info['year'] = int(year_match.group(1))
                
                if 'Fall' in text or 'Autumn' in text:
                    info['semester'] = 'Fall'
                elif 'Spring' in text:
                    info['semester'] = 'Spring'
            
            # Check if it's in Elective Courses section
            if text and ('Elective' in text or 'elective' in text.lower()):
                info['type'] = 'Elective'
            elif text and ('Mandatory' in text or 'Compulsory' in text):
                info['type'] = 'Mandatory'
        
        # Extract ECTS and local credits from row cells
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            text = cell.get_text(strip=True)
            # Look for ECTS value (usually last column)
            if text.isdigit() and len(text) <= 2:
                # Check if it's in ECTS column
                header = cell.find_previous(['th'])
                if header:
                    header_text = header.get_text(strip=True)
                    if 'ECTS' in header_text:
                        info['ects'] = int(text) if text.isdigit() else None
                    elif 'Local' in header_text or 'Credit' in header_text:
                        info['local_credits'] = int(text) if text.isdigit() else None
        
        return info
    
    def scrape_course_detail(self, course_url: str, course_code: str, scraped_course_codes: set = None) -> Dict:
        """
        Scrape detailed information from a course detail page
        Uses specific HTML IDs and table structures from IUE ECTS website
        
        Returns dict with course details
        """
        soup = self.fetch_page(course_url)
        if not soup:
            return {}
        
        course_detail = {
            'course_code': self.normalize_course_code(course_code),  # Görsel: Normalize
            'course_name': '',
            'objectives': '',
            'description': '',
            'semester': '',  # Fall/Spring
            'prerequisites': '',
            'weekly_topics': [],
            'learning_outcomes': [],  # Added back
            'type': '',  # Mandatory/Elective
            'theory_hours': None,
            'application_hours': None,  # Combined Application/Lab hours
            'local_credits': None,
            'ects': None,
            'assessment': {},  # Added back
            'ects_workload': {}  # Added back
        }
        
        try:
            # Method 1: Extract by specific HTML IDs (most reliable)
            # Course Name - from id="course_name"
            course_name_elem = soup.find(id='course_name')
            if course_name_elem:
                course_detail['course_name'] = course_name_elem.get_text(strip=True)
            
            # Prerequisites - from id="pre_requisites"
            prereq_elem = soup.find(id='pre_requisites')
            if prereq_elem:
                course_detail['prerequisites'] = prereq_elem.get_text(strip=True)
            
            # Theory hours - from id="weekly_hours"
            theory_elem = soup.find(id='weekly_hours')
            if theory_elem:
                try:
                    course_detail['theory_hours'] = int(theory_elem.get_text(strip=True))
                except (ValueError, AttributeError):
                    pass
            
            # Application/Lab hours - from id="app_hours" (combines both application and lab)
            app_elem = soup.find(id='app_hours')
            if app_elem:
                try:
                    app_hours = int(app_elem.get_text(strip=True))
                    course_detail['application_hours'] = app_hours
                except (ValueError, AttributeError):
                    pass
            
            # Semester - from id="semester"
            semester_elem = soup.find(id='semester')
            if semester_elem:
                semester_text = semester_elem.get_text(strip=True)
                if semester_text:
                    course_detail['semester'] = semester_text
            
            # Course Type (Mandatory/Elective) - from id="course_type"
            course_type_elem = soup.find(id='course_type')
            if course_type_elem:
                course_type_text = course_type_elem.get_text(strip=True)
                # Convert "Required" to "Mandatory", "Elective" stays as is
                if 'required' in course_type_text.lower():
                    course_detail['type'] = 'Mandatory'
                elif 'elective' in course_type_text.lower():
                    course_detail['type'] = 'Elective'
                else:
                    course_detail['type'] = course_type_text
            
            # Local Credits - from id="ieu_credit"
            local_credit_elem = soup.find(id='ieu_credit')
            if local_credit_elem:
                try:
                    course_detail['local_credits'] = int(local_credit_elem.get_text(strip=True))
                except (ValueError, AttributeError):
                    pass
            
            # ECTS - from id="ects_credit"
            ects_elem = soup.find(id='ects_credit')
            if ects_elem:
                try:
                    course_detail['ects'] = int(ects_elem.get_text(strip=True))
                except (ValueError, AttributeError):
                    pass
            
            # Method 2: Extract Course Objectives from table
            # Look for table row with "Course Objectives" label
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        if 'course objective' in label and not course_detail['objectives']:
                            course_detail['objectives'] = value
                        
                        elif 'course description' in label and not course_detail['description']:
                            course_detail['description'] = value
            
            # Learning Outcomes removed - not in requirements
            
            # Method 4: Extract Weekly Topics from table id="weeks" - ALL weeks
            # Table structure: Week | Subjects | Required Materials
            # Rows have id="hafta_X" format
            weeks_table = soup.find(id='weeks')
            if weeks_table:
                # Find tbody if exists, otherwise use table directly
                tbody = weeks_table.find('tbody')
                table_to_search = tbody if tbody else weeks_table
                
                rows = table_to_search.find_all('tr')
                logger.debug(f"Found {len(rows)} rows in weeks table")
                
                for row in rows:
                    # Skip header row (has class="table_top" or contains "Week" in first cell)
                    row_classes = row.get('class', [])
                    first_cell_text = ''
                    if row.find(['td', 'th']):
                        first_cell_text = row.find(['td', 'th']).get_text(strip=True).lower()
                    
                    if 'table_top' in row_classes or 'week' in first_cell_text:
                        logger.debug(f"Skipping header row: {first_cell_text}")
                        continue
                    
                    # Check if row has id="hafta_X" format (data row)
                    row_id = row.get('id', '')
                    
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        # Week number might be in <strong> tag or directly in cell
                        week_cell = cells[0]
                        week_text = week_cell.get_text(strip=True)
                        
                        # Extract week number (could be in <strong> tag)
                        week_match = re.search(r'\d+', week_text)
                        if week_match:
                            week_num = week_match.group(0)
                            
                            # Get topic from second cell - use get_text with separator to preserve spaces
                            topic_cell = cells[1]
                            # Use get_text without separator first, then clean up
                            topic = topic_cell.get_text(separator=' ', strip=False) if len(cells) > 1 else ''
                            # Replace newlines and multiple spaces with single space
                            topic = re.sub(r'[\n\r\t]+', ' ', topic)
                            topic = re.sub(r'\s+', ' ', topic).strip()
                            
                            # Get required materials from third cell - preserve all text including numbers
                            required_materials_cell = cells[2] if len(cells) > 2 else None
                            if required_materials_cell:
                                # Use get_text without separator to preserve all content
                                required_materials = required_materials_cell.get_text(separator=' ', strip=False)
                                # Replace newlines and multiple spaces with single space
                                required_materials = re.sub(r'[\n\r\t]+', ' ', required_materials)
                                required_materials = re.sub(r'\s+', ' ', required_materials).strip()
                            else:
                                required_materials = ''
                            
                            # Always add if we have a valid week number
                            if week_num:
                                course_detail['weekly_topics'].append({
                                    'week': week_num,
                                    'topic': topic if topic else '',
                                    'required_materials': required_materials if required_materials else ''
                                })
                                logger.debug(f"Added week {week_num}: {topic[:60] if topic else '(empty)'}")
            
            # Method 5: Extract Learning Outcomes from id="outcome"
            # Learning outcomes are in <ul id="outcome"><li> elements
            outcome_elem = soup.find(id='outcome')
            if outcome_elem:
                outcome_items = outcome_elem.find_all('li')
                for item in outcome_items:
                    # Get full text - use get_text with separator to preserve all text
                    # Don't use strip=True initially to preserve all content
                    outcome_text = item.get_text(separator=' ', strip=False)
                    # Replace newlines, tabs, and multiple spaces with single space
                    outcome_text = re.sub(r'[\n\r\t]+', ' ', outcome_text)
                    outcome_text = re.sub(r'\s+', ' ', outcome_text).strip()
                    if outcome_text:
                        course_detail['learning_outcomes'].append(outcome_text)
                        logger.debug(f"Added learning outcome: {outcome_text[:100]}")
            
            # Method 6: Extract Assessment/Evaluation System from id="evaluation_table1" and "evaluation_table2"
            eval_table1 = soup.find(id='evaluation_table1')
            if eval_table1:
                assessment_items = []
                rows = eval_table1.find_all('tr')
                for row in rows[1:-1]:  # Skip header and total row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        activity = cells[0].get_text(strip=True)
                        # Try to find div with class="editinput" or any div
                        number_elem = cells[1].find('div', class_='editinput') or cells[1].find('div')
                        weighting_elem = cells[2].find('div', class_='editinput') or cells[2].find('div')
                        
                        # If no div, get text directly
                        if not number_elem:
                            number_elem = cells[1]
                        if not weighting_elem:
                            weighting_elem = cells[2]
                        
                        number = number_elem.get_text(strip=True) if number_elem else '-'
                        weighting = weighting_elem.get_text(strip=True) if weighting_elem else '-'
                        
                        if activity and activity != 'Total':
                            assessment_items.append({
                                'activity': activity,
                                'number': number,
                                'weighting': weighting
                            })
                
                course_detail['assessment']['semester_activities'] = assessment_items
                
                # Get total from id="ara_total_no" and id="ara_total_per"
                total_no_elem = soup.find(id='ara_total_no')
                total_per_elem = soup.find(id='ara_total_per')
                
                if total_no_elem and total_per_elem:
                    total_number = total_no_elem.get_text(strip=True)
                    total_weighting = total_per_elem.get_text(strip=True)
                else:
                    # Fallback to last row
                    total_row = rows[-1] if rows else None
                    if total_row:
                        total_cells = total_row.find_all(['td', 'th'])
                        if len(total_cells) >= 3:
                            total_no = total_cells[1].find('div', class_='editinput') or total_cells[1].find('div')
                            total_per = total_cells[2].find('div', class_='editinput') or total_cells[2].find('div')
                            total_number = total_no.get_text(strip=True) if total_no else total_cells[1].get_text(strip=True)
                            total_weighting = total_per.get_text(strip=True) if total_per else total_cells[2].get_text(strip=True)
                        else:
                            total_number = ''
                            total_weighting = ''
                    else:
                        total_number = ''
                        total_weighting = ''
                
                course_detail['assessment']['total'] = {
                    'number': total_number,
                    'weighting': total_weighting
                }
            
            eval_table2 = soup.find(id='evaluation_table2')
            if eval_table2:
                rows = eval_table2.find_all('tr')
                weighting_info = {
                    'semester_activities': {},
                    'end_of_semester_activities': {},
                    'total': {}
                }
                
                for row in rows[:-1]:  # Skip total row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        label = cells[0].get_text(strip=True)
                        
                        # Get number from second cell - div is direct child
                        number_cell = cells[1]
                        number_div = number_cell.find('div', recursive=False)  # Only direct children
                        if number_div:
                            number = number_div.get_text(strip=True)
                        else:
                            number = number_cell.get_text(strip=True)
                        
                        # Get value from third cell - div is direct child
                        value_cell = cells[2]
                        value_div = value_cell.find('div', recursive=False)  # Only direct children
                        if value_div:
                            value = value_div.get_text(strip=True)
                        else:
                            value = value_cell.get_text(strip=True)
                        
                        # Normalize label and store in structured format
                        label_lower = label.lower()
                        # Check for End-of-Semester FIRST (more specific) to avoid matching "End-of-Semester Activities" with "semester activities" condition
                        if ('end-of-semester' in label_lower or 'end of semester' in label_lower) and 'final grade' in label_lower:
                            weighting_info['end_of_semester_activities'] = {
                                'label': label,
                                'number': number,
                                'value': value,
                                'percentage': int(value) if value and value.isdigit() else None
                            }
                            logger.debug(f"Found end-of-semester activities weighting: {number} activities, {value}%")
                        elif 'semester activities' in label_lower and 'final grade' in label_lower and 'end-of-semester' not in label_lower and 'end of semester' not in label_lower:
                            weighting_info['semester_activities'] = {
                                'label': label,
                                'number': number,
                                'value': value,
                                'percentage': int(value) if value and value.isdigit() else None
                            }
                            logger.debug(f"Found semester activities weighting: {number} activities, {value}%")
                
                # Get total - try id="total_no" and id="total_per" first, but they may be empty (JS-filled)
                # If empty, calculate from semester_activities + end_of_semester_activities
                total_no_elem = soup.find(id='total_no')
                total_per_elem = soup.find(id='total_per')
                
                total_number = ''
                total_value = ''
                
                if total_no_elem:
                    total_number = total_no_elem.get_text(strip=True)
                if total_per_elem:
                    total_value = total_per_elem.get_text(strip=True)
                
                # If empty, calculate from the two categories
                if not total_number or total_number == '':
                    sa_num = weighting_info.get('semester_activities', {}).get('number', '')
                    esa_num = weighting_info.get('end_of_semester_activities', {}).get('number', '')
                    if sa_num.isdigit() and esa_num.isdigit():
                        total_number = str(int(sa_num) + int(esa_num))
                        logger.debug(f"Calculated total number: {total_number} = {sa_num} + {esa_num}")
                
                if not total_value or total_value == '':
                    sa_pct = weighting_info.get('semester_activities', {}).get('percentage')
                    esa_pct = weighting_info.get('end_of_semester_activities', {}).get('percentage')
                    if sa_pct is not None and esa_pct is not None:
                        total_value = str(sa_pct + esa_pct)
                        logger.debug(f"Calculated total percentage: {total_value} = {sa_pct} + {esa_pct}")
                
                logger.debug(f"Total weighting: {total_number} activities, {total_value}%")
                
                weighting_info['total'] = {
                    'number': total_number,
                    'value': total_value,
                    'percentage': int(total_value) if total_value and total_value.isdigit() else None
                }
                
                course_detail['assessment']['weighting'] = weighting_info
            
            # Method 7: Extract ECTS Workload from id="workload_table"
            # Use specific IDs for each workload value (more reliable)
            workload_table = soup.find(id='workload_table')
            if workload_table:
                workload_items = []
                # Find tbody if exists
                tbody = workload_table.find('tbody')
                table_to_search = tbody if tbody else workload_table
                
                rows = table_to_search.find_all('tr')
                for row in rows[1:-1]:  # Skip header and total row
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 4:
                        # Activity name - preserve line breaks in activity name
                        activity_cell = cells[0]
                        activity = activity_cell.get_text(separator=' ', strip=False)
                        activity = re.sub(r'[\n\r\t]+', ' ', activity)
                        activity = re.sub(r'\s+', ' ', activity).strip()
                        
                        # Try to find specific IDs for this row's workload values
                        # Map activity names to ID patterns
                        activity_lower = activity.lower()
                        number_id = None
                        duration_id = None
                        workload_id = None
                        
                        if 'course hours' in activity_lower:
                            number_id = 'course_hour_number'
                            duration_id = 'course_hour_duration'
                            workload_id = 'course_hour_total_workload'
                        elif 'laboratory' in activity_lower or 'application hours' in activity_lower:
                            number_id = 'lab_number'
                            duration_id = 'lab_duration'
                            workload_id = 'lab_total_workload'
                        elif 'study hours out of class' in activity_lower:
                            number_id = 'out_hour_number'
                            duration_id = 'out_hour_duration'
                            workload_id = 'out_hour_total_workload'
                        elif 'field work' in activity_lower:
                            number_id = 'fieldwork_number'
                            duration_id = 'fieldwork_duration'
                            workload_id = 'fieldwork_total_number'
                        elif 'quizzes' in activity_lower or 'studio critiques' in activity_lower:
                            number_id = 'quizess_number'
                            duration_id = 'quizess_duration'
                            workload_id = 'quizess_total_workload'
                        elif 'portfolio' in activity_lower:
                            number_id = 'portfolioMed_number'
                            duration_id = 'portfolioMed_duration'
                            workload_id = 'portfolioMed_total_workload'
                        elif 'homework' in activity_lower or 'assignments' in activity_lower:
                            number_id = 'homework_number'
                            duration_id = 'homework_duration'
                            workload_id = 'homework_total_workload'
                        elif 'presentation' in activity_lower or 'jury' in activity_lower:
                            number_id = 'presentation_number'
                            duration_id = 'presentation_duration'
                            workload_id = 'presentation_total_workload'
                        elif 'project' in activity_lower:
                            number_id = 'project_number'
                            duration_id = 'project_duration'
                            workload_id = 'project_total_workload'
                        elif 'seminar' in activity_lower or 'workshop' in activity_lower:
                            number_id = 'seminar_number'
                            duration_id = 'seminar_duration'
                            workload_id = 'seminar_total_workload'
                        elif 'oral exam' in activity_lower:
                            number_id = 'portfolios_number'
                            duration_id = 'portfolios_duration'
                            workload_id = 'portfolios_total_workload'
                        elif 'midterm' in activity_lower:
                            number_id = 'midterm_number'
                            duration_id = 'midterm_duration'
                            workload_id = 'midterm_total_workload'
                        elif 'final exam' in activity_lower or 'final exams' in activity_lower:
                            number_id = 'final_number'
                            duration_id = 'final_duration'
                            workload_id = 'final_total_workload'
                        
                        # Get values from IDs if found, otherwise from cells
                        if number_id:
                            number_elem = soup.find(id=number_id)
                            number = number_elem.get_text(strip=True) if number_elem else '-'
                        else:
                            # Fallback: try to find div in cell
                            number_elem = cells[1].find('div', class_='editinput') or cells[1].find('div', recursive=False)
                            if number_elem:
                                number = number_elem.get_text(strip=True)
                            else:
                                number = cells[1].get_text(strip=True) if len(cells) > 1 else '-'
                        
                        if duration_id:
                            duration_elem = soup.find(id=duration_id)
                            duration = duration_elem.get_text(strip=True) if duration_elem else '-'
                        else:
                            # Fallback: try to find div in cell
                            duration_elem = cells[2].find('div', class_='editinput') or cells[2].find('div', recursive=False)
                            if duration_elem:
                                duration = duration_elem.get_text(strip=True)
                            else:
                                duration = cells[2].get_text(strip=True) if len(cells) > 2 else '-'
                        
                        if workload_id:
                            workload_elem = soup.find(id=workload_id)
                            if workload_elem:
                                workload = workload_elem.get_text(strip=True)
                                # If empty, try to calculate from number * duration
                                if not workload or workload == '':
                                    if number != '-' and duration != '-' and number.isdigit() and duration.isdigit():
                                        try:
                                            workload = str(int(number) * int(duration))
                                            logger.debug(f"Calculated workload for {activity[:40]}: {workload} = {number} * {duration}")
                                        except:
                                            workload = '-'
                                    else:
                                        workload = '-'
                                logger.debug(f"Found workload for {activity[:40]}: {workload} (from id={workload_id})")
                            else:
                                workload = '-'
                        else:
                            # Fallback: try to find div in cell
                            workload_elem = cells[3].find('div', class_='editinput') or cells[3].find('div', recursive=False)
                            if workload_elem:
                                workload = workload_elem.get_text(strip=True)
                                # If empty, try to calculate
                                if not workload or workload == '':
                                    if number != '-' and duration != '-' and number.isdigit() and duration.isdigit():
                                        try:
                                            workload = str(int(number) * int(duration))
                                        except:
                                            workload = '-'
                            else:
                                workload = cells[3].get_text(strip=True) if len(cells) > 3 else '-'
                                # If empty, try to calculate
                                if not workload or workload == '':
                                    if number != '-' and duration != '-' and number.isdigit() and duration.isdigit():
                                        try:
                                            workload = str(int(number) * int(duration))
                                        except:
                                            workload = '-'
                        
                        if activity:
                            workload_items.append({
                                'activity': activity,
                                'number': number,
                                'duration_hours': duration,
                                'workload': workload
                            })
                
                course_detail['ects_workload']['activities'] = workload_items
                
                # Get total workload - try to find div with class="editinput" in last row's last cell
                total_row = rows[-1] if rows else None
                if total_row:
                    total_cells = total_row.find_all(['td', 'th'])
                    if len(total_cells) >= 4:
                        total_cell = total_cells[-1]  # Last cell
                        total_elem = total_cell.find('div', class_='editinput') or total_cell.find('div')
                        if total_elem:
                            total_workload = total_elem.get_text(strip=True)
                        else:
                            total_workload = total_cell.get_text(strip=True)
                    else:
                        total_workload = ''
                else:
                    total_workload = ''
                
                course_detail['ects_workload']['total'] = total_workload
            
            
            # Fallback methods if IDs not found
            # Fallback 1: Course Name from table with "Course Name" label
            if not course_detail['course_name']:
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True).lower()
                            if 'course name' in label:
                                course_detail['course_name'] = cells[1].get_text(strip=True)
                                break
                    if course_detail['course_name']:
                        break
            
            # Fallback 3: Semester from table if ID not found
            if not course_detail['semester']:
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True).lower()
                            if 'semester' in label:
                                semester_text = cells[1].get_text(strip=True)
                                if semester_text:
                                    course_detail['semester'] = semester_text
                                break
            
            # Fallback 4: Course Type from table if ID not found
            if not course_detail['type']:
                for table in soup.find_all('table'):
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            label = cells[0].get_text(strip=True).lower()
                            if 'course type' in label:
                                type_text = cells[1].get_text(strip=True)
                                if 'required' in type_text.lower():
                                    course_detail['type'] = 'Mandatory'
                                elif 'elective' in type_text.lower():
                                    course_detail['type'] = 'Elective'
                                else:
                                    course_detail['type'] = type_text
                                break
            
            # Clean up text fields
            for key in ['objectives', 'description', 'prerequisites', 'course_name', 'semester', 'type']:
                if course_detail[key]:
                    # Remove extra whitespace and newlines
                    course_detail[key] = re.sub(r'\s+', ' ', course_detail[key]).strip()
                if not course_detail[key]:
                    course_detail[key] = ''
            
        except Exception as e:
            logger.error(f"Error scraping course detail for {course_code}: {e}")
            import traceback
            logger.error(traceback.format_exc())  # Changed to error level to see traceback
        
        # For POOL courses, fetch nested courses from pool page
        if course_code.startswith('POOL'):
            course_detail['available_courses'] = []
            course_detail['minimum_ects'] = None
            
            # Extract section from URL
            section_match = re.search(r'section=([^&]+)', course_url)
            section = section_match.group(1) if section_match else ''
            
            if section:
                # Find department URL
                dept_url = None
                for dept_key, dept_info in self.DEPARTMENTS.items():
                    if dept_info['section'] == section:
                        dept_url = dept_info['url']
                        break
                
                if dept_url:
                    # Fetch curriculum page to find POOL link
                    curriculum_soup = self.fetch_page(dept_url)
                    if curriculum_soup:
                        # Find POOL link and get semester info from its parent row
                        pool_link = None
                        pool_semester = None
                        all_links = curriculum_soup.find_all('a', href=True)
                        for link in all_links:
                            link_text = link.get_text(strip=True)
                            if link_text == course_code:
                                href = link.get('href', '')
                                if 'sid=pool' in href or 'pool' in href.lower():
                                    pool_link = link
                                    # Get semester from parent row
                                    parent_row = link.find_parent('tr')
                                    if parent_row:
                                        semester_info = self._extract_semester_info(parent_row)
                                        pool_semester = semester_info.get('semester')
                                    break
                        
                        if pool_link:
                            href = pool_link.get('href', '')
                            if not href.startswith('http'):
                                href = urljoin(self.BASE_URL, href)
                            
                            # Ensure section is in URL
                            if 'section=' not in href:
                                href = f"{href}&section={section}"
                            
                            # Fetch pool page
                            pool_soup = self.fetch_page(href)
                            if pool_soup:
                                pool_data = self._extract_pool_courses(pool_soup, section, course_code, pool_semester, scraped_course_codes)
                                if pool_data:
                                    course_detail['available_courses'] = pool_data.get('available_courses', [])
                                    course_detail['minimum_ects'] = pool_data.get('minimum_ects')
                                    course_detail['description'] = pool_data.get('description', '')
                                    if pool_semester:
                                        course_detail['description'] += f" Pool is taken in {pool_semester} semester."
        
        # For ELEC courses, fetch nested courses from curriculum page
        if course_code.startswith('ELEC'):
            course_detail['available_courses'] = []
            
            # Extract section from URL
            section_match = re.search(r'section=([^&]+)', course_url)
            section = section_match.group(1) if section_match else ''
            
            if section:
                # Find department URL
                dept_url = None
                for dept_key, dept_info in self.DEPARTMENTS.items():
                    if dept_info['section'] == section:
                        dept_url = dept_info['url']
                        break
                
                if dept_url:
                    # Fetch curriculum page
                    curriculum_soup = self.fetch_page(dept_url)
                    if curriculum_soup:
                        # Extract elective courses from table
                        elective_courses = self._extract_elective_courses_table(curriculum_soup, section)
                        
                        # Get ELEC parent semester from course_detail
                        elec_semester = course_detail.get('semester', '')
                        
                        # Filter electives by semester match
                        matching_electives = []
                        for elective in elective_courses:
                            elective_code = elective.get('course_code', '')
                            
                            # Skip if already scraped
                            if scraped_course_codes and elective_code in scraped_course_codes:
                                logger.debug(f"Skipping duplicate elective course: {elective_code}")
                                continue
                            
                            # Fetch semester from detail page
                            if elective.get('detail_url'):
                                elective_detail = self.scrape_course_detail(elective['detail_url'], elective_code, scraped_course_codes)
                                if elective_detail:
                                    # Mark as scraped
                                    if scraped_course_codes is not None:
                                        scraped_course_codes.add(elective_code)
                                    elective['semester'] = elective_detail.get('semester', '')
                                    
                                    # Check semester match
                                    elective_semester = elective.get('semester', '')
                                    should_include = False
                                    
                                    if not elec_semester:
                                        should_include = True
                                    elif elective_semester:
                                        elec_sem_lower = elec_semester.lower()
                                        elective_sem_lower = elective_semester.lower()
                                        
                                        if elec_sem_lower == elective_sem_lower:
                                            should_include = True
                                        elif 'fall' in elec_sem_lower and 'fall' in elective_sem_lower:
                                            should_include = True
                                        elif 'spring' in elec_sem_lower and 'spring' in elective_sem_lower:
                                            should_include = True
                                        elif 'fall/spring' in elec_sem_lower or 'fall/spring' in elective_sem_lower:
                                            should_include = True
                                    
                                    if should_include:
                                        # Add full detail
                                        elective.update({
                                            'objectives': elective_detail.get('objectives', ''),
                                            'description': elective_detail.get('description', ''),
                                            'weekly_topics': elective_detail.get('weekly_topics', []),
                                            'learning_outcomes': elective_detail.get('learning_outcomes', []),
                                            'assessment': elective_detail.get('assessment', {}),
                                            'ects_workload': elective_detail.get('ects_workload', {}),
                                            'prerequisites': elective_detail.get('prerequisites', '')
                                        })
                                        matching_electives.append(elective)
                        
                        course_detail['available_courses'] = matching_electives
                        course_detail['description'] = f"This elective course group contains {len(matching_electives)} available courses."
                        if elec_semester:
                            course_detail['description'] += f" Semester: {elec_semester}."
        
        return course_detail
    
    def scrape_department(self, dept_key: str, global_scraped_course_codes: set = None) -> List[Dict]:
        """
        Scrape all courses for a department
        
        Args:
            dept_key: Key from DEPARTMENTS dict
            global_scraped_course_codes: Set of course codes scraped across all departments (for cross-department duplicate prevention)
            
        Returns:
            List of course dictionaries with all information
        """
        if dept_key not in self.DEPARTMENTS:
            logger.error(f"Unknown department: {dept_key}")
            return []
        
        if global_scraped_course_codes is None:
            global_scraped_course_codes = set()
        
        dept_info = self.DEPARTMENTS[dept_key]
        logger.info(f"Scraping department: {dept_info['name']}")
        
        # Fetch curriculum page
        soup = self.fetch_page(dept_info['url'])
        if not soup:
            logger.error(f"Failed to fetch curriculum page for {dept_info['name']}")
            return []
        
        # Track scraped course codes to avoid duplicates (create early to share with extraction)
        # Use global set for cross-department tracking
        scraped_course_codes = global_scraped_course_codes
        
        # Extract course links (pass scraped_course_codes to track nested courses)
        courses = self.extract_course_links_from_curriculum(soup, dept_info['section'], scraped_course_codes)
        
        # Scrape details for each course
        all_courses = []
        for i, course in enumerate(courses, 1):
            raw_course_code = course.get('course_code', '')
            course_code = self.normalize_course_code(raw_course_code)  # Görsel: Normalize
            course['course_code'] = course_code  # Update in course dict
            
            # SFL courses are the same across all departments - skip if already scraped
            if course_code.startswith('SFL'):
                if course_code in scraped_course_codes:
                    logger.info(f"Skipping duplicate SFL course (already scraped in another department): {course_code}")
                    continue
                scraped_course_codes.add(course_code)
            
            # Skip if already scraped (for non-SFL/ELEC/POOL courses)
            elif not (course_code.startswith('ELEC') or course_code.startswith('POOL')):
                if course_code in scraped_course_codes:
                    logger.info(f"Skipping duplicate course: {course_code}")
                    continue
                scraped_course_codes.add(course_code)
            
            logger.info(f"Scraping course {i}/{len(courses)}: {course_code}")
            
            # POOL courses don't have detail pages - skip scraping detail
            detail = {}
            if course_code.startswith('POOL'):
                # POOL courses are already extracted with nested courses
                detail = {}
            elif course.get('detail_url'):
                detail = self.scrape_course_detail(
                    course['detail_url'],
                    course_code,
                    scraped_course_codes
                )
            
            # Merge curriculum info with detail info
            merged_course = {**course, **detail}
            # Override with detail info if available (detail page is more accurate)
            if detail.get('course_name'):
                merged_course['course_name'] = detail['course_name']
            if detail.get('semester'):
                merged_course['semester'] = detail['semester']
            if detail.get('type'):
                merged_course['type'] = detail['type']
            if detail.get('ects'):
                merged_course['ects'] = detail['ects']
            if detail.get('local_credits'):
                merged_course['local_credits'] = detail['local_credits']
            if detail.get('theory_hours'):
                merged_course['theory_hours'] = detail['theory_hours']
            if detail.get('application_hours'):
                merged_course['application_hours'] = detail['application_hours']
            
            # Process nested courses for SFL/ELEC/POOL (check for duplicates across all departments)
            if merged_course.get('available_courses'):
                filtered_nested = []
                for nested in merged_course['available_courses']:
                    nested_code = nested.get('course_code', '')
                    if nested_code:
                        # Check if already scraped (cross-department check)
                        if nested_code in scraped_course_codes:
                            logger.debug(f"Skipping duplicate nested course (already scraped): {nested_code}")
                            continue
                        
                        # Scrape nested course detail if URL is available
                        nested_url = nested.get('detail_url', '')
                        if nested_url:
                            nested_detail = self.scrape_course_detail(nested_url, nested_code, scraped_course_codes)
                            if nested_detail:
                                nested.update(nested_detail)
                        
                        # Mark as scraped (for cross-department duplicate prevention)
                        scraped_course_codes.add(nested_code)
                        filtered_nested.append(nested)
                merged_course['available_courses'] = filtered_nested
            
            # Remove fields that are not needed (not in requirements)
            merged_course.pop('learning_outcomes', None)
            merged_course.pop('assessment', None)
            merged_course.pop('ects_workload', None)
            merged_course.pop('laboratory_hours', None)
            
            merged_course['department'] = dept_info['name']
            merged_course['department_key'] = dept_key
            
            all_courses.append(merged_course)
        
        logger.info(f"Completed scraping {len(all_courses)} courses for {dept_info['name']}")
        return all_courses
    
    def scrape_all_departments(self) -> Dict[str, List[Dict]]:
        """Scrape all departments and return organized data"""
        all_data = {}
        
        # Global scraped course codes set to track across all departments
        # This prevents duplicate scraping of SFL, ELEC nested, and POOL nested courses
        global_scraped_course_codes = set()
        
        for dept_key in self.DEPARTMENTS.keys():
            courses = self.scrape_department(dept_key, global_scraped_course_codes)
            all_data[dept_key] = courses
        
        return all_data
    
    def save_to_json(self, data: Dict, output_path: str):
        """Save scraped data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {output_path}")


if __name__ == "__main__":
    scraper = IUECourseScraper(delay=1.0)
    
    # Scrape all departments
    all_data = scraper.scrape_all_departments()
    
    # Save to JSON
    output_path = "../data/raw/scraped_courses.json"
    scraper.save_to_json(all_data, output_path)
    
    # Print summary
    print("\n=== Scraping Summary ===")
    for dept_key, courses in all_data.items():
        print(f"{scraper.DEPARTMENTS[dept_key]['name']}: {len(courses)} courses")
