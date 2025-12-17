#!/usr/bin/env python3
"""
Test script to check SFL course detail page for nested courses
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def test_sfl_detail():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    course_code = "SFL 201"
    encoded_code = course_code.replace(" ", "%20")
    course_url = f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&currType=before_2025"
    
    print(f"Fetching SFL course detail page: {course_url}")
    soup = scraper.fetch_page(course_url)
    
    if not soup:
        print("Failed to fetch course detail page")
        return
    
    print("\n=== Checking for nested language courses in detail page ===")
    
    # Language course prefixes
    lang_prefixes = ['FR', 'ITL', 'GER', 'RUS', 'SPN', 'JPN', 'CHN', 'NFR', 'CFR', 'TFR']  # FR not TFR
    
    # Check all tables
    print("\n--- Checking tables ---")
    for i, table in enumerate(soup.find_all('table')):
        print(f"\nTable {i+1}:")
        rows = table.find_all('tr')
        for row in rows:
            lang_links = row.find_all('a', href=re.compile(r'syllabus\.php'))
            for link in lang_links:
                code = link.get_text(strip=True)
                if any(code.startswith(prefix) for prefix in lang_prefixes):
                    print(f"  Found language course: {code}")
                    cells = row.find_all(['td', 'th'])
                    for j, cell in enumerate(cells):
                        if link in cell.find_all('a'):
                            if j + 1 < len(cells):
                                name = cells[j + 1].get_text(strip=True)
                                print(f"    Name: {name}")
                            break
    
    # Check all divs
    print("\n--- Checking divs ---")
    for div in soup.find_all('div'):
        lang_links = div.find_all('a', href=re.compile(r'syllabus\.php'))
        if lang_links:
            print(f"Div with class: {div.get('class', [])}")
            for link in lang_links:
                code = link.get_text(strip=True)
                if any(code.startswith(prefix) for prefix in lang_prefixes):
                    print(f"  Found language course: {code}")
    
    # Check for JavaScript data attributes
    print("\n--- Checking for data attributes ---")
    for elem in soup.find_all(attrs={'data-code': True}):
        print(f"Element with data-code: {elem.get('data-code')}")
        print(f"  Text: {elem.get_text(strip=True)[:100]}")
    
    # Check for hidden elements
    print("\n--- Checking for hidden elements ---")
    for elem in soup.find_all(attrs={'style': re.compile(r'display.*none', re.I)}):
        text = elem.get_text(strip=True)
        if any(lang in text.upper() for lang in lang_prefixes):
            print(f"Hidden element with language courses: {text[:200]}")
    
    # Print full HTML to see structure
    print("\n=== Full HTML (first 2000 chars) ===")
    print(soup.prettify()[:2000])

if __name__ == "__main__":
    test_sfl_detail()
