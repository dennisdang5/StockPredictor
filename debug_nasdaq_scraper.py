#!/usr/bin/env python3
"""
Debug script to check Nasdaq page structure and identify why scraping is failing.
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

def get_webdriver():
    """Create a webdriver instance"""
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    # Don't run headless for debugging
    # options.add_argument("--headless")
    
    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print(f"Error creating webdriver: {e}")
        print("\nMake sure ChromeDriver is installed and in your PATH")
        print("Or install webdriver-manager: pip install webdriver-manager")
        return None

def debug_nasdaq_page(ticker="aaau"):
    """Debug a Nasdaq news headlines page"""
    url = f"https://www.nasdaq.com/market-activity/stocks/{ticker}/news-headlines"
    print(f"Testing URL: {url}")
    print("=" * 80)
    
    driver = get_webdriver()
    if not driver:
        return
    
    try:
        print("Loading page...")
        driver.get(url)
        time.sleep(5)  # Give page time to load
        
        print("\n1. Checking page title:")
        print(f"   Title: {driver.title}")
        
        print("\n2. Checking for logo element:")
        try:
            logo = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'nsdq-logo--default'))
            )
            print("   ✓ Logo found")
        except TimeoutException:
            print("   ✗ Logo not found")
            # Try alternative selectors
            try:
                logo_alt = driver.find_element(By.CSS_SELECTOR, '[class*="logo"]')
                print(f"   Found alternative logo: {logo_alt.get_attribute('class')}")
            except:
                print("   No logo elements found")
        
        print("\n3. Checking for pagination elements:")
        try:
            pagination = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'pagination__page'))
            )
            print(f"   ✓ Found {len(pagination)} pagination elements")
        except TimeoutException:
            print("   ✗ Pagination elements not found with class 'pagination__page'")
            # Try to find any pagination-related elements
            try:
                pagination_alt = driver.find_elements(By.CSS_SELECTOR, '[class*="pagination"]')
                print(f"   Found {len(pagination_alt)} elements with 'pagination' in class")
                for elem in pagination_alt[:5]:
                    print(f"     - {elem.get_attribute('class')}")
            except:
                print("   No pagination elements found")
        
        print("\n4. Checking for headline elements:")
        try:
            headlines = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'quote-news-headlines__item'))
            )
            print(f"   ✓ Found {len(headlines)} headline elements")
        except TimeoutException:
            print("   ✗ Headline elements not found with class 'quote-news-headlines__item'")
            # Try alternative selectors
            try:
                headlines_alt = driver.find_elements(By.CSS_SELECTOR, '[class*="headline"], [class*="news"]')
                print(f"   Found {len(headlines_alt)} elements with 'headline' or 'news' in class")
                for elem in headlines_alt[:5]:
                    print(f"     - {elem.get_attribute('class')}")
            except:
                print("   No headline elements found")
        
        print("\n5. Checking page source for key terms:")
        page_source = driver.page_source.lower()
        keywords = ['pagination', 'headline', 'news', 'quote-news']
        for keyword in keywords:
            count = page_source.count(keyword)
            print(f"   '{keyword}': {count} occurrences")
        
        print("\n6. Checking for JavaScript errors or blocking:")
        logs = driver.get_log('browser')
        if logs:
            print(f"   Found {len(logs)} browser log entries")
            for log in logs[:5]:
                print(f"     - {log['level']}: {log['message'][:100]}")
        else:
            print("   No browser logs found")
        
        print("\n7. Current URL after load:")
        print(f"   {driver.current_url}")
        
        print("\n8. Page ready state:")
        ready_state = driver.execute_script("return document.readyState")
        print(f"   {ready_state}")
        
        # Save page source for inspection
        with open(f"nasdaq_page_source_{ticker}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"\n9. Page source saved to: nasdaq_page_source_{ticker}.html")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 80)
        input("Press Enter to close browser...")
        driver.quit()

if __name__ == "__main__":
    ticker = input("Enter ticker symbol to test (default: aaau): ").strip().lower() or "aaau"
    debug_nasdaq_page(ticker)

