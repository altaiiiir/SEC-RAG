"""
Main entry point for Streamlit application.
This file serves as the home page and redirects to the Chat page.
"""
import streamlit as st

st.set_page_config(page_title="SECRAG", page_icon="📊", layout="wide")

st.title("📊 SECRAG - SEC Filing Analysis")
st.markdown("""
Welcome to SECRAG, a RAG system for analyzing SEC EDGAR filings.

### Available Pages:
- **Chat**: Ask questions about SEC filings with AI-powered answers
- **RAG Debug**: View raw vector search results for debugging

Use the sidebar to navigate between pages.
""")

st.info("👈 Select a page from the sidebar to get started!")
