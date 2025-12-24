#!/usr/bin/env python3
"""
Debug script to inspect weeks table structure for problematic courses
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_weeks_table():
    scraper = IUECourseScraper(delay=1.0)
    
    # Problematic courses: IUE 100, ENG 101, FENG 101
    test_courses = [
        ("IUE 100", "se.cs.ieu.edu.tr"),
        ("ENG 101", "se.cs.ieu.edu.tr")
    ]
    
    for course_code, section in test_courses:
        # Test both English (default/implicit) and Turkish
        versions = [
            ("Default", f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={course_code.replace(' ', '%20')}&currType=before_2025"),
            ("SFL Section", f"https://ects.ieu.edu.tr/new/syllabus.php?section=sfl.ieu.edu.tr&course_code={course_code.replace(' ', '%20')}&currType=before_2025")
        ]
        
        for v_name, course_url in versions:
            print(f"\n\n==================================================")
            print(f"Fetching course detail page ({v_name}): {course_url}")
        soup = scraper.fetch_page(course_url)
        
        if not soup:
            print("Failed to fetch course detail page")
            continue
        
        print("\n=== Checking weeks table (id='weeks') ===")
        weeks_table = soup.find(id='weeks')
        
        if not weeks_table:
            print("Weeks table not found!")
            continue
        
        print(f"Found weeks table")
        rows = weeks_table.find_all('tr')
        print(f"Total rows: {len(rows)}")
        
        for i, row in enumerate(rows):
            print(f"\n--- Row {i} ---")
            
            cells = row.find_all(['td', 'th'])
            
            # Print raw HTML of specific cells to check for hidden elements or structure
            for j, cell in enumerate(cells):
                cell_text = cell.get_text(separator=' ', strip=True)
                print(f"  Cell {j}: '{cell_text[:100]}'")
                
                # Check for id attribute on the cell or children
                if cell.get('id'):
                    print(f"    Cell ID: {cell.get('id')}")
            
            # Check extraction logic
            if len(cells) >= 2:
                week_cell = cells[0]
                week_text = week_cell.get_text(strip=True)
                week_match = re.search(r'\d+', week_text)
                
                if week_match:
                    topic_cell = cells[1]
                    topic = topic_cell.get_text(separator=' ', strip=False)
                    topic = re.sub(r'[\n\r\t]+', ' ', topic)
                    topic = re.sub(r'\s+', ' ', topic).strip()
                    print(f"  -> EXTRACTED: Week {week_match.group(0)}: '{topic}'")

if __name__ == "__main__":
    debug_weeks_table()
