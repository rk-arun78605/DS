import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, parse_qs
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
base_url = "https://fairwayghana.com"
headers = {"User-Agent": "Mozilla/5.0"}
MAX_PAGES = 5
THREADS = 20  # Adjust for system/network
TIMEOUT = 20

def get_category_paths():
    """Extract all category path values dynamically from homepage menu"""
    print("🔍 Extracting category paths...")
    try:
        res = requests.get(base_url, headers=headers, timeout=TIMEOUT)
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select("a[href*='route=product/category&path=']")
        paths = set()

        for link in links:
            href = link.get("href")
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "path" in qs:
                path = qs["path"][0]
                if "_" in path or path.isdigit():
                    paths.add(path)

        print(f"✅ Found {len(paths)} unique category paths.")
        return sorted(paths)
    except Exception as e:
        print(f"❌ Failed to extract category paths - {e}")
        return []

def build_category_urls(paths):
    urls = []
    for path in paths:
        for page in range(1, MAX_PAGES + 1):
            url = f"{base_url}/index.php?route=product/category&path={path}&limit=100&page={page}"
            urls.append(url)
    return urls

def get_breadcrumb(soup):
    crumbs = soup.select('.breadcrumb a')
    return " > ".join(c.get_text(strip=True) for c in crumbs)

def scrape_page(url):
    try:
        res = requests.get(url, headers=headers, timeout=TIMEOUT)
        soup = BeautifulSoup(res.text, "html.parser")
        breadcrumb = get_breadcrumb(soup)

        products = []
        for box in soup.select(".product-layout"):
            name = box.select_one(".caption a")
            price = box.select_one(".price")
            products.append({
                "Product": name.get_text(strip=True) if name else "N/A",
                "Price": price.get_text(strip=True) if price else "N/A",
                "Breadcrumb": breadcrumb,
                "Competitor": "Fareway",
                "URL": url
            })

        return products
    except Exception as e:
        print(f"❌ Error scraping {url} - {e}")
        return []

def run_fareway_scraper():
    print(f"🚀 Fareway scraper started: {datetime.datetime.now()}")
    start = time.time()

    paths = get_category_paths()
    urls = build_category_urls(paths)
    all_data = []

    print(f"🔄 Scraping {len(urls)} pages using {THREADS} threads...")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(scrape_page, url): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            all_data.extend(result)
            print(f"[{i}/{len(urls)}] ✅ Scraped {len(result)} items")

    df = pd.DataFrame(all_data)
    df.to_excel("Fareway_Products_Output.xlsx", index=False)

    print(f"\n📦 Output saved to Fareway_Products_Output.xlsx ({len(all_data)} items)")
    print(f"🏁 Finished in {int(time.time() - start)} seconds")

# ---------- MAIN ----------
if __name__ == "__main__":
    run_fareway_scraper()
