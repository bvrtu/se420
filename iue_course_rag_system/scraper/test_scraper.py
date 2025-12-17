"""
Test script for the IUE Course Scraper
Tests scraping functionality on a single department or course
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper import IUECourseScraper
from urllib.parse import urljoin
import json
import re

def test_single_department():
    """Test scraping a single department"""
    scraper = IUECourseScraper(delay=1.0)
    
    # Test with Software Engineering
    print("Testing Software Engineering department...")
    courses = scraper.scrape_department("software_engineering")
    
    print(f"\nScraped {len(courses)} courses")
    
    # Show first course as example
    if courses:
        print("\n=== Example Course ===")
        example = courses[0]
        for key, value in example.items():
            if key != 'weekly_topics':
                print(f"{key}: {value}")
            else:
                print(f"{key}: {len(value)} topics")
    
    # Save to file
    output_path = "../data/raw/test_software_engineering.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scraper.save_to_json({"software_engineering": courses}, output_path)
    
    return courses

def test_single_course(course_code=None, course_url=None, section=None):
    """
    Test scraping a single course detail page
    For SFL/ELEC/POOL courses, also extracts nested courses from curriculum
    
    Args:
        course_code: Course code (e.g., "SE 115", "CE 221", "SFL 201")
        course_url: Full URL to course detail page (optional)
        section: Department section (e.g., "se.cs.ieu.edu.tr") - used if URL not provided
    """
    scraper = IUECourseScraper(delay=1.0)
    
    # If URL not provided, construct it from course code and section
    if not course_url:
        if not course_code:
            # Default test
            course_code = "SE 115"
            section = "se.cs.ieu.edu.tr"
        
        if not section:
            # Try to guess section from course code prefix
            code_prefix = course_code.split()[0] if course_code else "SE"
            section_map = {
                "SE": "se.cs.ieu.edu.tr",
                "CE": "ce.cs.ieu.edu.tr",
                "EEE": "ete.cs.ieu.edu.tr",
                "IE": "is.cs.ieu.edu.tr",
                "SFL": "se.cs.ieu.edu.tr",  # Default to SE for SFL
                "ELEC": "se.cs.ieu.edu.tr",
                "POOL": "se.cs.ieu.edu.tr"
            }
            section = section_map.get(code_prefix, "se.cs.ieu.edu.tr")
        
        # Construct URL (use cer=0&sem=1 format without currType for proper page loading)
        encoded_code = course_code.replace(" ", "%20")
        course_url = f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1"
    
    print(f"Testing course detail scraping: {course_code}")
    print(f"URL: {course_url}\n")
    
    detail = scraper.scrape_course_detail(course_url, course_code)
    
    # For SFL/ELEC/POOL courses, also get nested courses from curriculum
    if course_code.startswith('SFL') or course_code.startswith('ELEC') or course_code.startswith('POOL'):
        print("This is a SFL/ELEC/POOL course. Fetching curriculum page to extract nested courses...")
        
        # Get department URL
        dept_url = None
        for dept_key, dept_info in scraper.DEPARTMENTS.items():
            if dept_info['section'] == section:
                dept_url = dept_info['url']
                break
        
        if dept_url:
            curriculum_soup = scraper.fetch_page(dept_url)
            if curriculum_soup:
                # Find the SFL/ELEC/POOL link in curriculum
                all_links = curriculum_soup.find_all('a', href=True)
                for link in all_links:
                    link_code = link.get_text(strip=True)
                    if link_code == course_code:
                        # Extract nested courses
                        if course_code.startswith('SFL'):
                            print(f"  Extracting nested language courses for {course_code}...")
                            nested = scraper.extract_sfl_courses_from_curriculum(link, curriculum_soup, section)
                            print(f"  Found {len(nested)} nested courses from curriculum extraction")
                            
                            if not nested:
                                print(f"  Trying alternative method: searching all language courses by pattern...")
                                # Alternative: Find all language courses matching SFL number pattern
                                lang_prefixes = ['FR', 'ITL', 'GER', 'RUS', 'SPN', 'JPN', 'CHN', 'NFR', 'CFR', 'TFR']  # FR not TFR
                                sfl_num = course_code.split()[-1] if ' ' in course_code else ''
                                
                                all_lang_courses = []
                                for lang_link in curriculum_soup.find_all('a', href=re.compile(r'syllabus\.php')):
                                    code = lang_link.get_text(strip=True)
                                    code_clean = code.replace(' ', '').upper()
                                    
                                    # Check if it's a language course
                                    is_lang = any(code.upper().startswith(p) for p in lang_prefixes)
                                    
                                    # Check if it matches SFL number pattern (more flexible)
                                    matches_pattern = False
                                    if sfl_num:
                                        sfl_num_clean = sfl_num.strip()
                                        # Try different patterns: "TFR 201", "TFR201", "TFR 2013", etc.
                                        if sfl_num_clean in code_clean or code_clean.endswith(sfl_num_clean):
                                            matches_pattern = True
                                    
                                    if is_lang and matches_pattern:
                                        parent_row = lang_link.find_parent('tr')
                                        name = ""
                                        if parent_row:
                                            cells = parent_row.find_all(['td', 'th'])
                                            for i, cell in enumerate(cells):
                                                if lang_link in cell.find_all('a'):
                                                    if i + 1 < len(cells):
                                                        name = cells[i + 1].get_text(strip=True)
                                                    break
                                        href = lang_link.get('href', '')
                                        if not href.startswith('http'):
                                            href = urljoin(scraper.BASE_URL, href)
                                        all_lang_courses.append({
                                            'course_code': code,
                                            'course_name': name,
                                            'detail_url': href
                                        })
                                
                                print(f"  Found {len(all_lang_courses)} language courses matching pattern *{sfl_num}")
                                nested = all_lang_courses
                            
                            if nested:
                                detail['available_courses'] = nested
                                lang_codes = ', '.join([c['course_code'] for c in nested])
                                detail['description'] = f"Second Foreign Languages course. Available language options: {lang_codes}"
                            else:
                                print(f"  Warning: Could not find nested courses for {course_code}")
                        elif course_code.startswith('POOL'):
                            # POOL courses are handled differently
                            href = link.get('href', '')
                            if 'sid=pool' in href or 'pool' in href.lower():
                                if not href.startswith('http'):
                                    href = urljoin(scraper.BASE_URL, href)
                                
                                # Get semester from parent row
                                pool_semester = None
                                parent_row = link.find_parent('tr')
                                if parent_row:
                                    semester_info = scraper._extract_semester_info(parent_row)
                                    pool_semester = semester_info.get('semester')
                                
                                pool_soup = scraper.fetch_page(href)
                                if pool_soup:
                                    pool_data = scraper._extract_pool_courses(pool_soup, section, course_code, pool_semester)
                                    if pool_data:
                                        detail['available_courses'] = pool_data.get('available_courses', [])
                                        detail['minimum_ects'] = pool_data.get('minimum_ects')
                                        detail['description'] = pool_data.get('description', '')
                                        if pool_semester:
                                            detail['description'] += f" Pool is taken in {pool_semester} semester."
                        elif course_code.startswith('ELEC'):
                            # ELEC courses: extract from elective courses table and match by semester
                            print(f"  Extracting nested elective courses for {course_code}...")
                            elective_courses = scraper._extract_elective_courses_table(curriculum_soup, section)
                            
                            # Get ELEC parent semester
                            elec_semester = detail.get('semester', '')
                            
                            # Filter and fetch details for matching electives
                            matching_electives = []
                            for elective in elective_courses:
                                # Fetch semester from detail page
                                if elective.get('detail_url'):
                                    elective_detail = scraper.scrape_course_detail(elective['detail_url'], elective['course_code'])
                                    if elective_detail:
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
                            
                            print(f"  Found {len(matching_electives)} matching elective courses")
                            detail['available_courses'] = matching_electives
                            detail['description'] = f"This elective course group contains {len(matching_electives)} available courses."
                            if elec_semester:
                                detail['description'] += f" Semester: {elec_semester}."
                        break
    
    print("\n=== Course Detail ===")
    for key, value in detail.items():
        if key == 'weekly_topics':
            print(f"{key}: {len(value)} topics")
            for topic in value:  # Show all topics
                topic_text = topic.get('topic', '')
                required_materials = topic.get('required_materials', '')
                if topic_text:
                    print(f"  Week {topic.get('week', '?')}: {topic_text}")
                    if required_materials:
                        print(f"    Required Materials: {required_materials}")
                else:
                    print(f"  Week {topic.get('week', '?')}: (empty)")
        elif key == 'learning_outcomes':
            print(f"{key}: {len(value)} outcomes")
            for i, outcome in enumerate(value, 1):  # Show all outcomes
                print(f"  {i}. {outcome}")
        elif key == 'assessment':
            print(f"{key}:")
            if isinstance(value, dict):
                if 'semester_activities' in value:
                    print(f"  Semester Activities: {len(value['semester_activities'])} items")
                    for item in value['semester_activities']:
                        activity = item.get('activity', '?')
                        number = item.get('number', '?')
                        weighting = item.get('weighting', '?')
                        if number != '-' or weighting != '-':
                            print(f"    - {activity}: {number} ({weighting}%)")
                if 'total' in value:
                    total = value['total']
                    if total.get('number') or total.get('weighting'):
                        print(f"  Total: {total.get('number', '?')} activities, {total.get('weighting', '?')}%")
                if 'weighting' in value:
                    weighting = value['weighting']
                    print(f"  Weighting:")
                    if 'semester_activities' in weighting and weighting['semester_activities']:
                        sa = weighting['semester_activities']
                        num = sa.get('number', '?')
                        pct = sa.get('percentage', '?')
                        print(f"    Semester Activities: {num} activities, {pct}%")
                    if 'end_of_semester_activities' in weighting and weighting['end_of_semester_activities']:
                        esa = weighting['end_of_semester_activities']
                        num = esa.get('number', '?')
                        pct = esa.get('percentage', '?')
                        print(f"    End-of-Semester Activities: {num} activities, {pct}%")
                    if 'total' in weighting and weighting['total']:
                        total = weighting['total']
                        num = total.get('number', '?')
                        pct = total.get('percentage', '?')
                        print(f"    Total: {num} activities, {pct}%")
        elif key == 'ects_workload':
            print(f"{key}:")
            if isinstance(value, dict):
                if 'activities' in value:
                    print(f"  Activities: {len(value['activities'])} items")
                    for item in value['activities']:
                        activity = item.get('activity', '?')
                        number = item.get('number', '?')
                        duration = item.get('duration_hours', '?')
                        workload = item.get('workload', '?')
                        # Clean up activity name (remove extra text in parentheses)
                        activity_clean = activity.split('(')[0].strip() if '(' in activity else activity
                        print(f"    - {activity_clean}: {number} x {duration}h = {workload}")
                if 'total' in value:
                    print(f"  Total Workload: {value['total']} hours")
        elif key == 'available_courses':
            print(f"{key}: {len(value)} nested courses")
            for nested in value[:10]:  # Show first 10
                print(f"  - {nested.get('course_code', '?')}: {nested.get('course_name', '')[:60]}")
            if len(value) > 10:
                print(f"  ... and {len(value) - 10} more")
        elif key == 'minimum_ects':
            if value:
                print(f"{key}: {value} ECTS (minimum requirement)")
        else:
            if value:  # Only print non-empty values
                print(f"{key}: {value}")
            else:
                print(f"{key}: (empty)")
    
    return detail

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test IUE Course Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test default course (SE 115)
  python test_scraper.py --test course
  
  # Test specific course by code
  python test_scraper.py --course "CE 221"
  
  # Test specific course with full URL
  python test_scraper.py --url "https://ects.ieu.edu.tr/new/syllabus.php?section=ce.cs.ieu.edu.tr&course_code=CE%20221&currType=before_2025"
  
  # Test specific course with code and section
  python test_scraper.py --course "IE 251" --section "is.cs.ieu.edu.tr"
        """
    )
    parser.add_argument("--test", choices=["department", "course", "all"], 
                       default="course", help="What to test")
    parser.add_argument("--course", type=str, 
                       help="Course code to test (e.g., 'SE 115', 'CE 221')")
    parser.add_argument("--url", type=str,
                       help="Full URL to course detail page")
    parser.add_argument("--section", type=str,
                       help="Department section (e.g., 'se.cs.ieu.edu.tr', 'ce.cs.ieu.edu.tr')")
    
    args = parser.parse_args()
    
    if args.test == "department" or args.test == "all":
        test_single_department()
    
    if args.test == "course" or args.test == "all":
        test_single_course(
            course_code=args.course,
            course_url=args.url,
            section=args.section
        )
