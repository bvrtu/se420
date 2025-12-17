#!/usr/bin/env python3
"""
Debug script to inspect assessment and workload tables
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_assessment_workload():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    course_code = "FR 103"
    encoded_code = course_code.replace(" ", "%20")
    course_url = f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1"
    
    print(f"Fetching course detail page: {course_url}")
    soup = scraper.fetch_page(course_url)
    
    if not soup:
        print("Failed to fetch course detail page")
        return
    
    print("\n=== Checking Evaluation Table 2 (Weighting) ===")
    eval_table2 = soup.find(id='evaluation_table2')
    if eval_table2:
        rows = eval_table2.find_all('tr')
        print(f"Found {len(rows)} rows")
        for i, row in enumerate(rows):
            print(f"\n--- Row {i} ---")
            cells = row.find_all(['td', 'th'])
            print(f"Number of cells: {len(cells)}")
            for j, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                divs = cell.find_all('div')
                print(f"  Cell {j}: '{cell_text[:60]}'")
                if divs:
                    for k, div in enumerate(divs):
                        div_id = div.get('id', '')
                        print(f"    Div {k}: '{div.get_text(strip=True)}' (id={div_id}, class={div.get('class', [])})")
    
    # Check for total_no and total_per IDs
    print("\n=== Checking Total IDs ===")
    total_no = soup.find(id='total_no')
    total_per = soup.find(id='total_per')
    print(f"total_no: {total_no.get_text(strip=True) if total_no else 'NOT FOUND'}")
    print(f"total_per: {total_per.get_text(strip=True) if total_per else 'NOT FOUND'}")
    
    # Check all elements with id containing 'total'
    print("\n=== All elements with 'total' in id ===")
    for elem in soup.find_all(id=re.compile('total', re.I)):
        print(f"  id='{elem.get('id')}': '{elem.get_text(strip=True)}'")
    
    print("\n=== Checking Workload Table ===")
    workload_table = soup.find(id='workload_table')
    if workload_table:
        tbody = workload_table.find('tbody')
        table_to_search = tbody if tbody else workload_table
        rows = table_to_search.find_all('tr')
        print(f"Found {len(rows)} rows")
        
        # Check specific IDs
        workload_ids = [
            'lab_total_workload',
            'quizess_total_workload',
            'homework_total_workload',
            'portfolios_total_workload',
            'midterm_total_workload',
            'final_total_workload'
        ]
        
        print("\n--- Checking specific workload IDs ---")
        for workload_id in workload_ids:
            elem = soup.find(id=workload_id)
            if elem:
                print(f"  {workload_id}: '{elem.get_text(strip=True)}'")
            else:
                print(f"  {workload_id}: NOT FOUND")
        
        # Check rows
        for i, row in enumerate(rows[1:4], 1):  # First 3 data rows
            print(f"\n--- Row {i} ---")
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 4:
                activity = cells[0].get_text(strip=True)[:50]
                print(f"  Activity: {activity}")
                for j in range(1, 4):
                    cell = cells[j]
                    cell_text = cell.get_text(strip=True)
                    divs = cell.find_all('div')
                    print(f"  Cell {j}: '{cell_text}'")
                    if divs:
                        for div in divs:
                            print(f"    Div: '{div.get_text(strip=True)}' (id={div.get('id', '')})")

if __name__ == "__main__":
    debug_assessment_workload()
