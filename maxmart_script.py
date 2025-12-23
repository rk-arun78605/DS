import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import time
import urllib3

# ---------- CONFIG ----------
base_url = "https://maxmartonline.com"
headers = {"User-Agent": "Mozilla/5.0"}
MAX_PAGES = 50
MAX_WORKERS = 25
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_category_slugs():
    """Extract top-level categories dynamically from homepage."""
    print("🔍 Extracting categories...")
    res = requests.get(base_url, headers=headers, verify=False, timeout=15)
    soup = BeautifulSoup(res.text, "html.parser")
    links = soup.select("a[href^='/']")
    slugs = {
        a['href'].strip("/") for a in links
        if a.get("href", "").count("/") == 1
        and not any(x in a['href'] for x in ["account", "login", "wishlist", "cart", "checkout", "blog"])
    }
    print(f"✅ Found {len(slugs)} categories.")
    return sorted(list(slugs))


def build_category_urls(slugs):
    """Create paginated product listing URLs for each category."""
    urls = []
    for slug in slugs:
        for page in range(1, MAX_PAGES + 1):
            url = f"{base_url}/{slug}?pagenumber={page}&viewmode=grid&orderby=0&pagesize=30"
            urls.append(url)
    return urls


def get_breadcrumb(soup):
    crumbs = soup.select(".breadcrumb li")
    return " > ".join(c.get_text(strip=True) for c in crumbs if c.get_text(strip=True))


def scrape_page(url):
    try:
        res = requests.get(url, headers=headers, timeout=15, verify=False)
        soup = BeautifulSoup(res.text, "html.parser")

        if not soup.select(".item-box"):
            return []  # no products on this page

        breadcrumb = get_breadcrumb(soup)
        products = []

        for box in soup.select(".item-box"):
            name = box.select_one(".product-title")
            price = box.select_one(".price")

            products.append({
                "Product": name.get_text(strip=True) if name else "N/A",
                "Price": price.get_text(strip=True) if price else "N/A",
                "Breadcrumb": breadcrumb,
                "Competitor": "MaxMart",
                "URL": url
            })
        return products
    except Exception as e:
        print(f"❌ Error scraping {url} - {e}")
        return []


def run_maxmart_scraper():
    print(f"🚀 MaxMart scraper started: {datetime.datetime.now()}")
    start = time.time()

    slugs = get_category_slugs()
    urls = build_category_urls(slugs)
    all_data = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_page, url): url for url in urls}
        for i, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                products = future.result()
                all_data.extend(products)
                print(f"[{i}/{len(futures)}] ✅ {url} ({len(products)} items)")
            except Exception as e:
                print(f"[{i}/{len(futures)}] ❌ {url} - {e}")

    df = pd.DataFrame(all_data)
    df.to_excel("MaxMart_Products_Output_Fast.xlsx", index=False)

    print(f"\n📦 Output saved to MaxMart_Products_Output_Fast.xlsx ({len(all_data)} items)")
    print(f"🏁 Finished in {int(time.time() - start)} seconds")


# ---------- MAIN ----------
if __name__ == "__main__":
    run_maxmart_scraper()
