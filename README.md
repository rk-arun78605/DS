# Data Science / Inventory Management Scripts

This repository contains scripts for inventory management and web scraping for retail operations.

## Files

### Inventory Management System
- **nowhstock_final.py** - Main inventory pulse application for NO_WH (No Warehouse) stock management
  - Built with Streamlit for interactive UI
  - Uses PostgreSQL for data storage
  - Provides stock transfer recommendations
  - Version 2.3 with dynamic sales calculations

- **mv_recommendations.sql** - PostgreSQL materialized view definition
  - Contains complete business logic for recommendations
  - Handles priority shop destinations
  - Implements sales calculations and capping logic
  - Includes GRN age calculations and blocking rules

### Web Scrapers
- **fareway.py** - Web scraper for Fairway Ghana website
  - Extracts product information from catalog pages
  - Uses concurrent processing for efficiency
  - Exports data to structured format

- **maxmart_script.py** - Script for MaxMart operations
  - Handles MaxMart-specific data processing

## Setup

### Requirements
- Python 3.x
- PostgreSQL database
- Required Python packages: streamlit, pandas, psycopg2, beautifulsoup4

### Database Configuration
**IMPORTANT**: Update database credentials in `nowhstock_final.py` before deploying. For security, use environment variables instead of hard-coded passwords:

```python
import os

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),  # Use environment variable
    'port': int(os.getenv('DB_PORT', 3307))
}
```

The current code contains a placeholder password that should be changed before production use.

## Usage

### Running the Inventory System
```bash
streamlit run nowhstock_final.py
```

### Running the Scrapers
```bash
python fareway.py
python maxmart_script.py
```

## Repository Cleanup (Dec 4, 2025)
This repository was cleaned up to remove duplicate and versioned files. Previous versions of files with timestamps and version suffixes in filenames have been removed. Version control should be managed through git commits rather than file duplication.
