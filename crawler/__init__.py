"""
crawler/__init__.py

Web crawling utilities: search APIs, faculty scraping, profile extraction.
"""

from crawler.search_api import search_professors, search_semantic_scholar_batch
from crawler.faculty_scraper import scrape_faculty_page, scrape_batch
from crawler.profile_extractor import ProfileExtractor

__all__ = [
    "search_professors",
    "search_semantic_scholar_batch",
    "scrape_faculty_page",
    "scrape_batch",
    "ProfileExtractor",
]
