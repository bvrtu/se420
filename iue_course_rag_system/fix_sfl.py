import json
import logging
from scraper.scraper import IUECourseScraper
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fix_sfl')

def fix_sfl_courses():
    json_path = "data/raw/scraped_courses.json"
    
    # Load existing data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("scraped_courses.json not found!")
        return

    scraper = IUECourseScraper()
    
    target_dept = 'software_engineering'
    if target_dept not in data:
        # Fallback to keys
        target_dept = list(data.keys())[0] if data else None
        
    if not target_dept:
        logger.error("No data found")
        return
        
    courses = data[target_dept]
    
    # Find SFL parents
    sfl_parents = [c for c in courses if c['course_code'].startswith('SFL')]
    
    logger.info(f"Found {len(sfl_parents)} SFL parent courses")
    
    new_courses = []
    
    # Keep track of added codes to avoid dups within this run
    added_codes = set()
    
    for parent in sfl_parents:
        parent_code = parent['course_code']
        parent_url = parent['detail_url']
        
        logger.info(f"Processing parent: {parent_code}")
        
        # 1. Scrape parent to get injected available_courses
        # We use empty set for scraped_codes to ensure we get results
        parent_detail = scraper.scrape_course_detail(parent_url, parent_code, set())
        
        children = parent_detail.get('available_courses', [])
        logger.info(f"  Found {len(children)} children for {parent_code}")
        
        for child in children:
            child_code = child['course_code']
            child_url = child['detail_url']
            
            # Check if already exists in courses, if so remove it to overwrite
            courses[:] = [c for c in courses if c['course_code'] != child_code]
                
            if child_code in added_codes:
                continue

            logger.info(f"  Scraping child: {child_code}")
            
            # Scrape child detail
            child_detail = scraper.scrape_course_detail(child_url, child_code)
            
            # Construct full course object
            # Merge child basic info + detail
            full_child = child.copy()
            full_child.update(child_detail)
            
            # Add department info (inherit from parent context)
            full_child['department'] = parent.get('department', 'Software Engineering')
            full_child['department_key'] = parent.get('department_key', 'software_engineering')
            full_child['parent_course'] = parent_code
            
            # Clean up
            full_child.pop('learning_outcomes', None)
            full_child.pop('assessment', None)
            full_child.pop('ects_workload', None)
            
            new_courses.append(full_child)
            added_codes.add(child_code)
            time.sleep(1.0) # Polite delay
            
    if new_courses:
        logger.info(f"Adding {len(new_courses)} new nested courses to dataset")
        data[target_dept].extend(new_courses)
        
        # Save back
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved updated scraped_courses.json")
    else:
        logger.info("No new courses to add.")

if __name__ == "__main__":
    fix_sfl_courses()
