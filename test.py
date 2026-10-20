"""Visible end-to-end demo: enter scenarios in Streamlit, click Predict, move on.

Run from the project folder: python test.py
Requires: python -m pip install playwright
Uses your installed Chrome by default; --bundled uses Playwright's Chromium.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parent
BASE = {
    "Number of dependents": 1,
    "Education": "Graduate",
    "Self-employed": "No",
    "Annual income (₹)": 8_000_000,
    "Loan amount (₹)": 5_000_000,
    "Loan term (years)": 10,
    "CIBIL score": 750,
    "Residential asset value (₹)": 10_000_000,
    "Commercial asset value (₹)": 5_000_000,
    "Luxury asset value (₹)": 8_000_000,
    "Bank asset value (₹)": 3_000_000,
}

# These expected labels were checked with the saved Random Forest generated
# from the supplied CSVs. The browser result is read from the live UI.
SCENARIOS = [
    ("1. Strong credit baseline", {}, "Approved"),
    ("2. Change only CIBIL: 750 → 450", {"CIBIL score": 450}, "Rejected"),
    ("3. Strong credit, few assets (ATR = 0.5)", {
        "Residential asset value (₹)": 1_000_000,
        "Commercial asset value (₹)": 500_000,
        "Luxury asset value (₹)": 500_000,
        "Bank asset value (₹)": 500_000,
    }, "Rejected"),
    ("4. Weak credit, short term, high loan/income and assets", {
        "CIBIL score": 450, "Loan term (years)": 2,
        "Annual income (₹)": 1_500_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Approved"),
    ("5. Same profile, longer loan term", {
        "CIBIL score": 450, "Loan term (years)": 12,
        "Annual income (₹)": 1_500_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Rejected"),
    ("6. Short term, but smaller loan/income ratio", {
        "CIBIL score": 450, "Loan term (years)": 2,
        "Annual income (₹)": 3_000_000, "Loan amount (₹)": 6_000_000,
        "Residential asset value (₹)": 15_000_000,
    }, "Rejected"),
    ("7. Change education, employment, dependents", {
        "Education": "Not Graduate", "Self-employed": "Yes",
        "Number of dependents": 5,
    }, "Approved"),
]


def server_ready(url):
    try:
        with urlopen(url.rstrip("/") + "/_stcore/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def fill_form(page, data):
    for label, value in data.items():
        if label in ("Education", "Self-employed"):
            page.get_by_label(label, exact=True).click()
            page.get_by_role("option", name=value, exact=True).click()
        else:
            page.get_by_label(label, exact=True).fill(str(value))


def main():
    parser = argparse.ArgumentParser(description="Demonstrate loan app scenarios in a visible browser")
    parser.add_argument("--url", default="http://localhost:8501", help="Existing Streamlit URL")
    parser.add_argument("--pause", type=float, default=5, help="Seconds to show each result")
    parser.add_argument("--headless", action="store_true", help="Run without opening a visible window")
    parser.add_argument("--bundled", action="store_true", help="Use Playwright Chromium instead of installed Chrome")
    args = parser.parse_args()
    if args.pause < 0:
        parser.error("--pause must be at least zero")
