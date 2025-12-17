#!/usr/bin/env python3
"""
Debug script to inspect weeks table structure
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_weeks_table():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    course_code = "FR 103"
    encoded_code = course_code.replace(" ", "%20")
    # Test with cer=0&sem=1 format (without currType)
    course_url = f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1"
    
    print(f"Fetching course detail page: {course_url}")
    soup = scraper.fetch_page(course_url)
    
    if not soup:
        print("Failed to fetch course detail page")
        return
    
    print("\n=== Checking weeks table (id='weeks') ===")
    weeks_table = soup.find(id='weeks')
    
    if not weeks_table:
        print("Weeks table not found!")
        return
    
    print(f"Found weeks table")
    rows = weeks_table.find_all('tr')
    print(f"Total rows: {len(rows)}")
    
    for i, row in enumerate(rows):
        print(f"\n--- Row {i} ---")
        print(f"Row classes: {row.get('class', [])}")
        print(f"Row id: {row.get('id', '')}")
        
        cells = row.find_all(['td', 'th'])
        print(f"Number of cells: {len(cells)}")
        
        for j, cell in enumerate(cells):
            cell_text = cell.get_text(strip=True)
            cell_html = str(cell)[:200]
            print(f"  Cell {j}: '{cell_text[:50]}'")
            print(f"    HTML: {cell_html}")
        
        # Check if this looks like a data row
        if i > 0:  # Skip header
            if len(cells) >= 2:
                week_text = cells[0].get_text(strip=True)
                topic_text = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                print(f"  -> Week: '{week_text}', Topic: '{topic_text[:60]}'")

if __name__ == "__main__":
    debug_weeks_table()
