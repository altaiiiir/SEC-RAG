import streamlit as st
import requests
import os
from typing import Optional

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SEC EDGAR RAG",
    page_icon="📊",
    layout="wide",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .result-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .similarity-score {
        color: #0066cc;
        font-weight: bold;
    }
    .metadata {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_stats():
    """Get database statistics."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def search_documents(query: str, top_k: int, ticker: Optional[str], filing_type: Optional[str]):
    """Search documents via API."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
        }
        if ticker:
            payload["ticker"] = ticker
        if filing_type:
            payload["filing_type"] = filing_type
            
        response = requests.post(f"{API_URL}/query", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        return None


# Main UI
st.title("📊 SEC EDGAR RAG System")
st.markdown("Search through SEC 10-K and 10-Q filings using AI-powered semantic search")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API health check
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Unavailable")
        st.stop()
    
    # Database stats
    stats = get_stats()
    if stats:
        st.metric("Documents", stats["total_documents"])
        st.metric("Chunks", f"{stats['total_chunks']:,}")
        st.metric("Companies", stats["total_tickers"])
        
        with st.expander("Filing Types"):
            for filing_type, count in stats.get("by_filing_type", {}).items():
                st.write(f"**{filing_type}**: {count} documents")
    
    st.divider()
    
    # Search filters
    st.subheader("🔍 Filters")
    
    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL",
        help="Filter by company ticker"
    ).upper().strip()
    
    filing_type = st.selectbox(
        "Filing Type",
        options=["All", "10-K", "10-Q"],
        help="Filter by filing type"
    )
    
    top_k = st.slider(
        "Results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of results to return"
    )

# Main search area
query = st.text_input(
    "🔎 Enter your question",
    placeholder="e.g., What was Apple's revenue in Q4 2024?",
    key="search_query"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Perform search
if search_button and query:
    with st.spinner("Searching documents..."):
        results = search_documents(
            query=query,
            top_k=top_k,
            ticker=ticker_input if ticker_input else None,
            filing_type=filing_type if filing_type != "All" else None
        )
    
    if results:
        st.success(f"Found {results['total_results']} results in {results['took_ms']:.0f}ms")
        
        # Display results
        for i, result in enumerate(results["results"], 1):
            with st.container():
                st.markdown(f"### Result {i}")
                
                # Metadata row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"**Ticker:** {result['ticker']}")
                with col2:
                    st.markdown(f"**Type:** {result['filing_type']}")
                with col3:
                    st.markdown(f"**Date:** {result['filing_date'] or 'N/A'}")
                with col4:
                    similarity_pct = result['similarity'] * 100
                    st.markdown(f"**Similarity:** <span class='similarity-score'>{similarity_pct:.1f}%</span>", unsafe_allow_html=True)
                
                # Content
                st.markdown("**Content:**")
                st.text_area(
                    label="",
                    value=result['content'],
                    height=150,
                    key=f"result_{i}",
                    label_visibility="collapsed"
                )
                
                # Document info
                st.caption(f"Document: {result['doc_id']} | Chunk ID: {result['chunk_id']}")
                
                st.divider()
    else:
        st.warning("No results found")

elif search_button and not query:
    st.warning("Please enter a search query")

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - What was Apple's revenue in 2024?
    - What are Tesla's main risk factors?
    - How much did Microsoft spend on R&D?
    - What is Amazon's operating income?
    - Describe Google's competitive landscape
    - What are Nvidia's AI chip capabilities?
    """)

# Footer
st.divider()
st.caption("SEC EDGAR RAG System | Built with FastAPI, Streamlit, and pgvector")
