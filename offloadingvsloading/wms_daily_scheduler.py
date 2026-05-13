"""
WMS Daily Scheduler — runs as a Windows Task Scheduler job.

Schedule this script to run once daily (e.g. 7:00 AM) with:
  "C:\Users\DIRECTOR CID\AppData\Local\Programs\Python\Python311\python.exe"  "d:\Dashboard Code\NO_WH\DS\offloadingvsloading\wms_daily_scheduler.py"

It checks if yesterday's data is already in the DB; if not, it syncs
from the network share and sends the update email.
"""

import logging
import os
import sys

# Ensure the offloadingvsloading folder is on the path when called externally
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wms_sync import run_sync

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wms_daily_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main():
    log.info("=== WMS daily scheduler starting ===")
    result = run_sync()
    log.info("Result: [%s] %s", result["status"].upper(), result["message"])
    if result["status"] == "error":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
