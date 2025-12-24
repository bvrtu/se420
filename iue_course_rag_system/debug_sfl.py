from scraper.scraper import IUECourseScraper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('scraper')

def debug_sfl():
    scraper = IUECourseScraper()
    # SFL 1013 URL from grep output
    url = "https://ects.ieu.edu.tr/syllabus.php?section=se.cs.ieu.edu.tr&course_code=SFL%201013&currType=before_2025"
    course_code = "SFL1013"
    
    print(f"Fetching details for {course_code}...")
    detail = scraper.scrape_course_detail(url, course_code)
    
    print(f"\nExtracted Available Courses: {len(detail.get('available_courses', []))}")
    for course in detail.get('available_courses', []):
        print(f"- {course.get('course_code')} : {course.get('course_name')}")

if __name__ == "__main__":
    debug_sfl()
