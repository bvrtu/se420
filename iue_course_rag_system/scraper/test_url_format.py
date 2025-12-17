#!/usr/bin/env python3
"""
Test different URL formats to see which one returns correct content
"""
import sys
from scraper import IUECourseScraper
from bs4 import BeautifulSoup
import re

def test_url_formats():
    scraper = IUECourseScraper(delay=1.0)
    section = "se.cs.ieu.edu.tr"
    course_code = "FR 103"
    encoded_code = course_code.replace(" ", "%20")
    
    # Test different URL formats
    urls = [
        f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1",
        f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&currType=before_2025",
        f"https://ects.ieu.edu.tr/new/syllabus.php?section={section}&course_code={encoded_code}&cer=0&sem=1&currType=before_2025",
    ]
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*80}")
        print(f"Testing URL format {i}: {url}")
        print('='*80)
        
        soup = scraper.fetch_page(url)
        if not soup:
            print("Failed to fetch")
            continue
        
        # Check weeks table
        weeks_table = soup.find(id='weeks')
        if weeks_table:
            rows = weeks_table.find_all('tr')
            print(f"Found {len(rows)} rows in weeks table")
            
            # Check first data row (after header)
            if len(rows) > 1:
                first_data_row = rows[1]
                cells = first_data_row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    week = cells[0].get_text(strip=True)
                    topic = cells[1].get_text(strip=True)
                    print(f"  Week {week}: {topic[:80]}")
                    
                    # Check if it's "Review of the Semester" or actual content
                    if "Review of the Semester" in topic:
                        print("  ❌ Contains 'Review of the Semester'")
                    else:
                        print("  ✅ Contains actual content")
        else:
            print("  Weeks table not found")
        
        # Check learning outcomes
        outcome_elem = soup.find(id='outcome')
        if outcome_elem:
            items = outcome_elem.find_all('li')
            print(f"Found {len(items)} learning outcomes")
            if items:
                first_outcome = items[0].get_text(separator=' ', strip=True)
                print(f"  First outcome: {first_outcome[:100]}")

if __name__ == "__main__":
    test_url_formats()
