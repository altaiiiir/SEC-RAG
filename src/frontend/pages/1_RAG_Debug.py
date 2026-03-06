import streamlit as st
import requests

from src.frontend.utils import API_URL, render_sidebar

st.set_page_config(page_title="RAG Debug", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .similarity-score {
        color: #0066cc;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def search_documents(query: str, top_k: int, ticker: str = None, filing_type: str = None, chunk_type: str = None):
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
        if chunk_type:
            payload["chunk_type"] = chunk_type
            
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
st.title("🔍 RAG Debug View")
st.markdown("Raw vector search results for debugging and testing")

# Sidebar
with st.sidebar:
    stats = render_sidebar()
    
    if stats:
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
    
    chunk_type_filter = st.selectbox(
        "Chunk Type",
        options=["All", "table", "list", "financial_statement", "narrative"],
        help="Filter by content type"
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
            filing_type=filing_type if filing_type != "All" else None,
            chunk_type=chunk_type_filter if chunk_type_filter != "All" else None
        )
    
    if results:
        st.success(f"Found {results['total_results']} results in {results['took_ms']:.0f}ms")
        
        # Display results
        for i, result in enumerate(results["results"], 1):
            with st.container():
                st.markdown(f"### Result {i}")
                
                # Metadata row with chunk type
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"**Ticker:** {result['ticker']}")
                with col2:
                    st.markdown(f"**Type:** {result['filing_type']}")
                with col3:
                    st.markdown(f"**Date:** {result['filing_date'] or 'N/A'}")
                with col4:
                    chunk_type = result.get('chunk_type', 'N/A')
                    st.markdown(f"**Chunk:** {chunk_type}")
                with col5:
                    similarity_pct = result['similarity'] * 100
                    st.markdown(f"**Similarity:** <span class='similarity-score'>{similarity_pct:.1f}%</span>", unsafe_allow_html=True)
                
                # Section and table metadata if available
                metadata_info = []
                if result.get('section_name'):
                    metadata_info.append(f"📄 Section: {result['section_name']}")
                if result.get('table_id'):
                    metadata_info.append(f"📊 Table: {result['table_id']}")
                if result.get('row_range'):
                    metadata_info.append(f"📏 Rows: {result['row_range']}")
                
                if metadata_info:
                    st.caption(" | ".join(metadata_info))
                
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

# Footer
st.divider()
