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

    # Clean LinkedIn metadata
    text = re.sub(r"\d+\+?\s*connections?\s*(on\s+LinkedIn)?\.?", "", text, flags=re.I)
    text = re.sub(r"View\s+[\w\s']+\s+profile\s+on\s+LinkedIn.*$", "", text, flags=re.I)
    text = re.sub(r"a\s+professional\s+community.*$", "", text, flags=re.I)

    # Convert "Location: City" to "located in City"
    text = re.sub(r"Location:\s*", "located in ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing ellipsis
    while text.endswith("..."):
        text = text[:-3].strip()

    # Truncate to last complete sentence
    if text and text[-1] not in ".!?":
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                text = text[: i + 1]
                break
        else:
            # No sentence ending found - add period
            text = text + "."

    return text


# =============================================================================
# Search Function
# =============================================================================
def search(query):
    """
    Search DuckDuckGo for information.

    Tries two strategies:
        1. Instant Answer API - structured JSON response
        2. HTML Scrape - parse search result snippets

    Args:
        query: Search query string

    Returns:
        str: First relevant search result, or empty string if no results
    """
    q = urllib.parse.quote(query)

    # Strategy 1: Try Instant Answer API
    try:
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            d = json.loads(r.read().decode())

        # Check for structured answers
        for k in ("AbstractText", "Answer", "Definition"):
            if d.get(k):
                return _clean(d[k])

        # Check related topics
        if d.get("RelatedTopics"):
            for t in d["RelatedTopics"]:
                if isinstance(t, dict) and t.get("Text"):
                    return _clean(t["Text"])
    except Exception:
        pass

    # Strategy 2: Fallback to HTML scrape
    try:
        url = f"https://html.duckduckgo.com/html/?q={q}"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            raw = r.read().decode("utf-8", errors="ignore")

        # Extract first result snippet
        m = re.search(r'<a class="result__snippet"[^>]*>(.+?)</a>', raw, re.DOTALL)
        if m:
            text = re.sub(r"<[^>]+>", "", m.group(1))
            return _clean(text)
    except Exception:
        pass

    return ""
