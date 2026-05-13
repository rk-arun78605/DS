"""
Gulfood 2026 Exhibitors Scraper - Production Version
Extracts: Company Name, Hall/Location, Country
Optimized for 6000+ pages
"""

import time
import csv
from datetime import datetime
import logging
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('gulfood_scrape.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

class GulfoodExhibitorScraper:
    def __init__(self, full_scrape=False):
        self.url = "https://exhibitors.gulfood.com/gulfood-2026/Exhibitors"
        self.driver = None
        self.full_scrape = full_scrape
        self.exhibitors = []
        self.csv_file = "Gulfood_Exhibitors_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def setup(self):
        """Initialize Chrome"""
        try:
            opts = Options()
            opts.add_argument('--start-maximized')
            opts.add_argument('--disable-blink-features=AutomationControlled')
            opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            svc = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=svc, options=opts)
            self.driver.set_page_load_timeout(30)
            
            log.info("SETUP OK")
            return True
        except Exception as e:
            log.error("SETUP FAILED: {}".format(e))
            return False
    
    def extract_from_page(self):
        """Extract all exhibitors from current page with full details"""
        try:
            time.sleep(2)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            page_exhibits = []
            
            # Find all h5 tags (company names) and extract related details
            for h5 in soup.find_all('h5'):
                name = h5.get_text(strip=True)
                if not name or len(name) < 2:
                    continue
                
                # Get the parent container (article or section)
                container = h5.find_parent(['article', 'section', 'div'])
                if not container:
                    container = h5.parent
                
                # Extract hall/location (look for link with map marker)
                hall_location = ""
                location_link = container.find('a', href=lambda x: x and 'map.gulfood.com' in x)
                if location_link:
                    hall_location = location_link.get_text(strip=True)
                
                # Extract country (look for span with country info)
                country = ""
                # Try to find span with country (usually after the name)
                all_spans = container.find_all('span')
                for span in all_spans:
                    span_text = span.get_text(strip=True)
                    # Country is typically 2-50 chars, single or double word
                    if 2 < len(span_text) < 50 and span_text not in [name, hall_location]:
                        if not any(char.isdigit() for char in span_text[:3]):  # Skip if starts with numbers
                            country = span_text
                            break
                
                # Also try to find country from p tag containing span
                if not country:
                    p_tags = container.find_all('p')
                    for p in p_tags:
                        spans = p.find_all('span')
                        for span in spans:
                            span_text = span.get_text(strip=True)
                            if 2 < len(span_text) < 50 and span_text not in [name, hall_location]:
                                country = span_text
                                break
                        if country:
                            break
                
                exhibitor = {
                    'Company_Name': name,
                    'Hall_Location': hall_location if hall_location else "N/A",
                    'Country': country if country else "N/A"
                }
                page_exhibits.append(exhibitor)
            
            log.info("PAGE EXTRACT: {} items (Company | Hall | Country)".format(len(page_exhibits)))
            if page_exhibits:
                log.info("SAMPLE: {} | {} | {}".format(
                    page_exhibits[0]['Company_Name'][:40],
                    page_exhibits[0]['Hall_Location'][:30],
                    page_exhibits[0]['Country'][:30]
                ))
            
            return page_exhibits
        
        except Exception as e:
            log.error("EXTRACT ERROR: {}".format(e))
            return []
    
    def next_page(self):
        """Navigate next"""
        try:
            time.sleep(1)
            buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), '»')]")
            if buttons:
                self.driver.execute_script("arguments[0].click();", buttons[0])
                time.sleep(3)
                return True
            return False
        except Exception as e:
            log.error("NEXT ERROR: {}".format(e))
            return False
    
    def run(self):
        """Main scrape"""
        try:
            log.info("START: Loading {}".format(self.url))
            self.driver.get(self.url)
            
            max_pages = 9999 if self.full_scrape else 5
            
            for page_num in range(1, max_pages + 1):
                log.info("PAGE {} | TOTAL: {} exhibitors".format(page_num, len(self.exhibitors)))
                
                items = self.extract_from_page()
                self.exhibitors.extend(items)
                
                # Save progress every 50 pages
                if page_num % 50 == 0:
                    self.save()
                    log.info("AUTO-SAVE at page {}".format(page_num))
                
                if page_num < max_pages:
                    if not self.next_page():
                        log.info("NO MORE PAGES at page {}".format(page_num))
                        break
            
            log.info("DONE: Scraped {} pages, {} total exhibitors".format(page_num, len(self.exhibitors)))
            return True
        
        except KeyboardInterrupt:
            log.info("INTERRUPTED by user")
            return False
        except Exception as e:
            log.error("RUN ERROR: {}".format(e))
            return False
    
    def save(self):
        """Save to CSV"""
        try:
            if not self.exhibitors:
                log.warning("SAVE: No data")
                return False
            
            with open(self.csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Company_Name', 'Hall_Location', 'Country'])
                writer.writeheader()
                writer.writerows(self.exhibitors)
            
            log.info("SAVE OK: {} items to {}".format(len(self.exhibitors), self.csv_file))
            return True
        
        except Exception as e:
            log.error("SAVE ERROR: {}".format(e))
            return False
    
    def cleanup(self):
        """Close driver"""
        if self.driver:
            try:
                self.driver.quit()
                log.info("CLEANUP OK")
            except:
                pass

# Main
if __name__ == "__main__":
    log.info("="*60)
    log.info("GULFOOD EXHIBITORS SCRAPER - PRODUCTION")
    log.info("Extracting: Company Name | Hall Location | Country")
    log.info("="*60)
    
    # Change to: full_scrape=True for all pages (6000+)
    # Change to: full_scrape=False for 5 pages (TEST)
    scraper = GulfoodExhibitorScraper(full_scrape=True)  # Set to True for full scrape
    
    try:
        if scraper.setup():
            scraper.run()
            scraper.save()
            log.info("="*60)
            log.info("SUCCESS! Output: {}".format(scraper.csv_file))
            log.info("="*60)
    finally:
        scraper.cleanup()
