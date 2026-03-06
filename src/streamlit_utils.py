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
    </style>
    """

def render_evidence_card(ev):
    """Render a single evidence card."""
    return f"""
    <div class="evidence-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <strong>{ev['ticker']}</strong> | {ev['filing_type']} | {ev['filing_date'] or 'N/A'}
            </div>
            <span class="similarity-badge">{ev['similarity']*100:.0f}% match</span>
        </div>
        <div style="font-size: 0.9rem; opacity: 0.8;">
            {ev['content'][:300]}...
        </div>
    </div>
    """
