"""
orchestrator/tools.py

Shared LangChain tools that agents can invoke via function calling.
"""

from langchain.tools import tool
from typing import Annotated, List, Dict, Any

from crawler.search_api import search_professors
from crawler.faculty_scraper import scrape_faculty_page
from core.cv_parser import CVParser

@tool
def search_tool(
    keywords: Annotated[List[str], "Keywords to search for"], 
    countries: Annotated[List[str], "Target countries to restrict the search to"]
) -> List[Dict[str, Any]]:
    """Search for relevant professors using Serper/Tavily."""
    return search_professors(keywords=keywords, countries=countries, max_per_query=5)

@tool
def scrape_tool(
    url: Annotated[str, "The URL of the professor's page or lab site to scrape"]
) -> str:
    """Scrape a web page and return its raw text content."""
    result = scrape_faculty_page(url)
    return result.get("raw_page_text", "") or ""

@tool
def parse_cv_tool(
    cv_path: Annotated[str, "The absolute path to the PDF CV file"]
) -> str:
    """Extract and summarize a user profile from a PDF CV."""
    parser = CVParser()
    summary, status = parser.extract_and_summarize(cv_path)
    if "Error" in status:
        return f"Failed to parse CV: {status}"
    return summary
