#!/usr/bin/env python3
"""
Debug script to inspect HTML structure around SFL courses
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_sfl_structure():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    dept_url = scraper.DEPARTMENTS['software_engineering']['url']
    
    print(f"Fetching curriculum page: {dept_url}")
    soup = scraper.fetch_page(dept_url)
    
    if not soup:
        print("Failed to fetch curriculum page")
        return
    
    # Find SFL 201 link
    print("\n=== Searching for SFL 201 link ===")
    all_links = soup.find_all('a', href=True)
    sfl_link = None
    for link in all_links:
        if link.get_text(strip=True) == 'SFL 201':
            sfl_link = link
            print(f"Found SFL 201 link: {link.get('href', '')}")
            break
    
    if not sfl_link:
        print("SFL 201 link not found!")
        # Try to find any SFL link
        for link in all_links:
            if 'SFL' in link.get_text(strip=True):
                print(f"Found SFL link: {link.get_text(strip=True)} -> {link.get('href', '')}")
        return
    
    # Find parent row
    parent_row = sfl_link.find_parent('tr')
    if parent_row:
        print("\n=== SFL 201 Parent Row ===")
        print(parent_row.prettify()[:500])
    
    # Find all language courses in the page
    print("\n=== All Language Courses in Page ===")
    lang_prefixes = ['FR', 'ITL', 'GER', 'RUS', 'SPN', 'JPN', 'CHN', 'NFR', 'CFR', 'TFR']  # FR not TFR
    lang_courses = []
    for link in soup.find_all('a', href=re.compile(r'syllabus\.php')):
        code = link.get_text(strip=True)
        if any(code.startswith(p) for p in lang_prefixes):
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
            lang_courses.append({
                'code': code,
                'name': name,
                'href': href
            })
    
    print(f"Found {len(lang_courses)} language courses:")
    for course in lang_courses[:20]:  # Show first 20
        print(f"  {course['code']}: {course['name'][:50]}")
    
    # Check for courses ending with 201
    print("\n=== Language Courses ending with 201 ===")
    matching = [c for c in lang_courses if c['code'].endswith('201')]
    print(f"Found {len(matching)} courses:")
    for course in matching:
        print(f"  {course['code']}: {course['name'][:50]}")
    
    # Check HTML structure around SFL 201
    print("\n=== HTML Structure Around SFL 201 ===")
    if parent_row:
        parent_table = parent_row.find_parent('table')
        if parent_table:
            all_rows = parent_table.find_all('tr')
            sfl_row_idx = None
            for idx, row in enumerate(all_rows):
                if parent_row == row:
                    sfl_row_idx = idx
                    break
            
            if sfl_row_idx is not None:
                print(f"SFL 201 is at row index {sfl_row_idx}")
                print("\nNext 10 rows after SFL 201:")
                for i, row in enumerate(all_rows[sfl_row_idx + 1:sfl_row_idx + 11]):
                    row_text = row.get_text(strip=True)
                    links = row.find_all('a', href=re.compile(r'syllabus\.php'))
                    link_codes = [l.get_text(strip=True) for l in links]
                    print(f"  Row {sfl_row_idx + 1 + i}: {row_text[:100]}")
                    if link_codes:
                        print(f"    Links: {', '.join(link_codes)}")

if __name__ == "__main__":
    debug_sfl_structure()
