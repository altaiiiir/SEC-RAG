"""Query parser to extract company tickers and section hints from natural language queries."""
import re
from typing import List, Optional, Dict


# Company name to ticker mapping (common companies in SEC filings)
COMPANY_TICKER_MAP = {
    # Tech
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "oracle": "ORCL",
    "salesforce": "CRM",
    
    # Finance
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "chase": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "berkshire hathaway": "BRK",
    "berkshire": "BRK",
    
    # Pharma
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "johnson and johnson": "JNJ",
    "merck": "MRK",
    "abbvie": "ABBV",
    "eli lilly": "LLY",
    "bristol myers": "BMY",
    
    # Retail & Consumer
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "home depot": "HD",
    "procter & gamble": "PG",
    "procter and gamble": "PG",
    "coca cola": "KO",
    "coca-cola": "KO",
    "pepsico": "PEP",
    "pepsi": "PEP",
    
    # Industrial
    "boeing": "BA",
    "caterpillar": "CAT",
    "deere": "DE",
    "john deere": "DE",
    "general electric": "GE",
    "3m": "MMM",
}

# Section name patterns (case-insensitive)
SECTION_PATTERNS = {
    "risk": ["Risk Factors", "Risks"],
    "business": ["Business"],
    "financial": ["Financial Statements and Supplementary Data", "Financial Data"],
    "mda": ["Management's Discussion and Analysis", "MD&A"],
    "compensation": ["Executive Compensation"],
    "controls": ["Controls and Procedures"],
    "legal": ["Legal Proceedings"],
    "properties": ["Properties"],
    "cybersecurity": ["Cybersecurity"],
}


def parse_query(query: str) -> Dict[str, any]:
    """
    Parse a natural language query to extract:
    - tickers: List of company tickers mentioned
    - section_hint: Suggested section name to filter by
    - original_query: The original query string
    
    Args:
        query: Natural language query
        
    Returns:
        Dictionary with parsed information
    """
    query_lower = query.lower()
    
    # Extract tickers
    tickers = []
    for company_name, ticker in COMPANY_TICKER_MAP.items():
        if company_name in query_lower:
            if ticker not in tickers:
                tickers.append(ticker)
    
    # Also check for direct ticker mentions (e.g., "AAPL", "TSLA")
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    direct_tickers = re.findall(ticker_pattern, query)
    for ticker in direct_tickers:
        if ticker not in tickers and len(ticker) <= 5:
            tickers.append(ticker)
    
    # Extract section hint
    section_hint = None
    for keyword, section_names in SECTION_PATTERNS.items():
        if keyword in query_lower:
            # Use the first (most specific) section name
            section_hint = section_names[0]
            break
    
    return {
        'tickers': tickers if tickers else None,
        'section_hint': section_hint,
        'original_query': query,
        'is_multi_company': len(tickers) > 1 if tickers else False,
    }


def suggest_top_k(parsed: Dict[str, any], default: int = 5) -> int:
    """
    Suggest appropriate top_k based on query complexity.
    
    Args:
        parsed: Parsed query dictionary
        default: Default top_k value
        
    Returns:
        Suggested top_k value
    """
    # Multi-company queries need more results
    if parsed.get('is_multi_company'):
        return max(15, default * len(parsed['tickers']))
    
    # Section-filtered queries can be more focused
    if parsed.get('section_hint'):
        return default
    
    return default


if __name__ == "__main__":
    # Test examples
    test_queries = [
        "What are the primary risk factors facing Apple, Tesla, and JPMorgan?",
        "Compare revenue growth for MSFT and GOOGL",
        "What is Amazon's cybersecurity strategy?",
        "How does Pfizer's R&D spending compare to Merck?",
    ]
    
    for query in test_queries:
        result = parse_query(query)
        suggested_k = suggest_top_k(result)
        print(f"\nQuery: {query}")
        print(f"Tickers: {result['tickers']}")
        print(f"Section: {result['section_hint']}")
        print(f"Multi-company: {result['is_multi_company']}")
        print(f"Suggested top_k: {suggested_k}")
