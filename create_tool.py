import os
from langchain_core.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv  
load_dotenv()
from rich import print

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_current_news(query: str) -> str:
    """search the web for a recent and reliable information on a topic and give me title URLs snippet and content"""
    response = tavily.search(query=query, topic="news")

    out = []
    results = response.get("results", [])
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        out.append(
            f"Title: {title}\nURL: {url}\nContent: {content}\n"
        )
    return "\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and extract the full text content from a given webpage URL. Use this to get detailed information from a URL returned by get_current_news."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script, style, and nav noise
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return clean_text[:5000]  # Limit output to avoid token overflow

    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"
    except Exception as e:
        return f"Error parsing content: {e}"



