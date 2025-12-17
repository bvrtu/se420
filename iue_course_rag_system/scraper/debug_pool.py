#!/usr/bin/env python3
"""
Debug script to inspect POOL course table structure
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def debug_pool():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    pool_code = "POOL 006"
    
    # Fetch pool page
    pool_url = f"https://ects.ieu.edu.tr/akademik.php?section={section}&sid=pool&currType=before_2025"
    print(f"Fetching pool page: {pool_url}")
    soup = scraper.fetch_page(pool_url)
    
    if not soup:
        print("Failed to fetch pool page")
        return
    
    print("\n=== Checking for POOL 006 heading ===")
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
    for heading in headings:
        heading_text = heading.get_text(strip=True)
        if 'POOL' in heading_text.upper() and '006' in heading_text:
            print(f"Found heading: {heading_text}")
    
    print("\n=== Checking all tables ===")
    tables = soup.find_all('table')
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        print(f"\n--- Table {i} ---")
        rows = table.find_all('tr')
        print(f"Rows: {len(rows)}")
        
        if len(rows) > 0:
            header_row = rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            print(f"Headers: {headers}")
            
            if len(rows) > 1:
                print(f"\nFirst data row:")
                first_data_row = rows[1]
                cells = first_data_row.find_all(['td', 'th'])
                for j, cell in enumerate(cells):
                    text = cell.get_text(strip=True)
                    links = cell.find_all('a', href=re.compile(r'syllabus\.php'))
                    print(f"  Cell {j}: '{text[:50]}' (links: {len(links)})")
                    if links:
                        for link in links:
                            print(f"    Link: {link.get_text(strip=True)} -> {link.get('href', '')[:60]}")

if __name__ == "__main__":
    debug_pool()
