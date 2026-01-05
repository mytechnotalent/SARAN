"""
Web Search Agent for SARAN

===============================================================================
DUCKDUCKGO WEB SEARCH AGENT
===============================================================================

This module provides web search capabilities for the SARAN chat interface.
It uses DuckDuckGo as the search backend with two fallback strategies:

1. Instant Answer API
   - Fast, structured responses
   - Returns AbstractText, Answer, Definition, or RelatedTopics
   - Best for factual queries

2. HTML Scrape Fallback
   - Parses search result snippets from HTML
   - Used when API returns no results
   - More reliable for general queries

Features:
    - HTML entity decoding (e.g., &#x27; -> ')
    - Non-ASCII character removal
    - Truncation to complete sentences
    - Ellipsis handling

===============================================================================
"""

import html
import json
import re
import ssl
import urllib.parse
import urllib.request

# Disable SSL verification (for environments with certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context

# =============================================================================
# Configuration
# =============================================================================
TIMEOUT = 8  # Request timeout in seconds
HEADERS = {"User-Agent": "Mozilla/5.0"}  # Browser user agent for requests


# =============================================================================
# Text Cleaning
# =============================================================================
def _clean(text):
    """
    Clean and normalize search result text.

    Processing steps:
        1. Decode HTML entities (&#x27; -> ')
        2. Remove non-ASCII characters
        3. Normalize whitespace
        4. Remove trailing ellipsis
        5. Truncate to last complete sentence

    Args:
        text: Raw text from search results

    Returns:
        str: Cleaned text ending with complete sentence
    """
    # Decode HTML entities
    text = html.unescape(text)

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing ellipsis
    while text.endswith("..."):
        text = text[:-3].strip()

    # Truncate to last complete sentence if needed
    if text and text[-1] not in ".!?":
        # Find last sentence-ending punctuation followed by space
        found = False
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1] == " "):
                text = text[: i + 1]
                found = True
                break
        if not found and text:
            text = text + "."  # Add period if no sentence ending found

    return text


# =============================================================================
# Search Function
# =============================================================================
def search(query, max_results=3):
    """
    Search DuckDuckGo and return combined relevant results.

    Tries two strategies:
        1. Instant Answer API - structured JSON response
        2. HTML Scrape - parse search result snippets

    Args:
        query: Search query string
        max_results: Maximum number of results to combine (default 3)

    Returns:
        str: Combined search results text, or empty string if no results
    """
    q = urllib.parse.quote(query)
    results = []

    # Strategy 1: Try Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode())

        # Check for structured answers (these are usually high quality)
        for k in ("AbstractText", "Answer", "Definition"):
            if d.get(k):
                cleaned = _clean(d[k])
                if cleaned and len(cleaned) > 20:
                    results.append(cleaned)

        # Check related topics for additional context
        if d.get("RelatedTopics"):
            for t in d["RelatedTopics"]:
                if isinstance(t, dict) and t.get("Text"):
                    cleaned = _clean(t["Text"])
                    if cleaned and len(cleaned) > 20:
                        results.append(cleaned)
                        if len(results) >= max_results:
                            break
    except Exception:
        pass

    # Strategy 2: Fallback to HTML scrape (always try for more results)
    try:
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", errors="ignore")

        # Extract multiple result snippets
        snippets = re.findall(
            r'<a class="result__snippet"[^>]*>(.+?)</a>', raw, re.DOTALL
        )
        for snippet in snippets[:max_results]:
            # Remove HTML tags from snippet
            text = re.sub(r"<[^>]+>", "", snippet)
            cleaned = _clean(text)
            if cleaned and len(cleaned) > 20 and cleaned not in results:
                results.append(cleaned)
                if len(results) >= max_results:
                    break
    except Exception:
        pass

    # Combine results into a single response
    if results:
        return " ".join(results)
    return ""
