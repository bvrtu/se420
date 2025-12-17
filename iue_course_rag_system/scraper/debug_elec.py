#!/usr/bin/env python3
"""
Debug script to inspect ELEC course table structure
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_elec():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    
    # Fetch curriculum page
    dept_url = None
    for dept_key, dept_info in scraper.DEPARTMENTS.items():
        if dept_info['section'] == section:
            dept_url = dept_info['url']
            break
    
    if not dept_url:
        print("Could not find department URL")
        return
    
    print(f"Fetching curriculum page: {dept_url}")
    soup = scraper.fetch_page(dept_url)
    
    if not soup:
        print("Failed to fetch curriculum page")
        return
    
    print("\n=== Checking all headings ===")
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
    for heading in headings:
        heading_text = heading.get_text(strip=True)
        if 'elective' in heading_text.lower() or 'course' in heading_text.lower():
            print(f"Found heading: {heading_text}")
    
    print("\n=== Checking all tables ===")
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        rows = table.find_all('tr')
        if len(rows) > 0:
            header_row = rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            header_text = ' '.join(headers).lower()
            
            if 'code' in header_text and ('course' in header_text or 'name' in header_text):
                print(f"\nTable {i}: Looks like a course table")
                print(f"  Headers: {headers}")
                print(f"  Rows: {len(rows)}")
                if len(rows) > 1:
                    print(f"  First data row:")
                    first_row = rows[1]
                    cells = first_row.find_all(['td', 'th'])
                    for j, cell in enumerate(cells):
                        text = cell.get_text(strip=True)
                        links = cell.find_all('a', href=re.compile(r'syllabus\.php'))
                        if links or text:
                            print(f"    Cell {j}: '{text[:40]}' (links: {len(links)})")
    
    print("\n=== Extracting elective courses ===")
    elective_courses = scraper._extract_elective_courses_table(soup, section)
    print(f"Found {len(elective_courses)} elective courses")
    
    for i, course in enumerate(elective_courses[:5], 1):  # Show first 5
        print(f"  {i}. {course.get('course_code')}: {course.get('course_name', '')[:50]}")

if __name__ == "__main__":
    debug_elec()
