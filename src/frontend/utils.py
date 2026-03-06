"""Shared utilities for Streamlit pages."""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

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

def render_sidebar():
    """Render common sidebar with API health and stats."""
    st.header("⚙️ Settings")
    
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Unavailable")
        st.stop()
    
    stats = get_stats()
    if stats:
        st.metric("Documents", stats["total_documents"])
        st.metric("Chunks", f"{stats['total_chunks']:,}")
        st.metric("Companies", stats["total_tickers"])
    
    return stats

def get_common_css():
    """Get common CSS styles."""
    return """
    <style>
        .evidence-card {
            background-color: transparent;
            padding: 0.8rem;
            border-radius: 0.3rem;
            margin-bottom: 0.5rem;
            border-left: 3px solid #0066cc;
            border: 1px solid rgba(0, 102, 204, 0.2);
        }
        .similarity-badge {
            background-color: #0066cc;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.3rem;
            font-size: 0.8rem;
            font-weight: bold;
        }
        .chunk-type-badge {
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.3rem;
            font-size: 0.7rem;
            font-weight: bold;
            text-transform: uppercase;
        }
    </style>
    """

def render_evidence_card(ev):
    """Render a single evidence card with metadata."""
    # Chunk type badge
    chunk_type = ev.get('chunk_type', 'narrative')
    chunk_type_colors = {
        'table': '#4CAF50',
        'list': '#FF9800',
        'financial_statement': '#9C27B0',
        'narrative': '#2196F3'
    }
    chunk_color = chunk_type_colors.get(chunk_type, '#2196F3')
    
    # Build metadata line
    metadata_parts = [f"<strong>{ev['ticker']}</strong>", ev['filing_type'], ev['filing_date'] or 'N/A']
    
    # Add section name if available
    if ev.get('section_name'):
        metadata_parts.append(f"<em>{ev['section_name']}</em>")
    
    # Add table/row info if available
    if ev.get('table_id') and ev.get('row_range'):
        metadata_parts.append(f"Rows: {ev['row_range']}")
    
    metadata_line = " | ".join(metadata_parts)
    
    return f"""
    <div class="evidence-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="flex-grow: 1;">
                {metadata_line}
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="chunk-type-badge" style="background-color: {chunk_color};">{chunk_type}</span>
                <span class="similarity-badge">{ev['similarity']*100:.0f}%</span>
            </div>
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8;">
            {ev['content'][:300]}...
        </div>
    </div>
    """
