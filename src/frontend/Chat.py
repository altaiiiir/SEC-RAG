import streamlit as st
import requests
import json

from src.frontend.utils import (
    API_URL, check_api_health, get_stats, render_sidebar, 
    get_common_css, render_evidence_card
)

st.set_page_config(page_title="SECRAG Chat", page_icon="💬", layout="wide")
st.markdown(get_common_css(), unsafe_allow_html=True)


def ask_question(query: str, top_k: int, ticker: str = None, filing_type: str = None):
    """Ask a question and get streaming response with evidence."""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
        }
        if ticker:
            payload["ticker"] = ticker
        if filing_type:
            payload["filing_type"] = filing_type
        
        response = requests.post(
            f"{API_URL}/ask",
            json=payload,
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            answer_text = ""
            evidence_data = []
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_json = line_str[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_json)
                            
                            if data['type'] == 'content':
                                answer_text += data['data']
                                yield {'type': 'content', 'data': data['data']}
                            elif data['type'] == 'evidence':
                                evidence_data = data['data']
                            elif data['type'] == 'done':
                                yield {'type': 'done', 'evidence': evidence_data}
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            yield {'type': 'error', 'message': f"API error: {response.status_code}"}
    
    except Exception as e:
        yield {'type': 'error', 'message': f"Request failed: {str(e)}"}


# Main UI
st.header("SECRAG Chat")

# Sidebar
with st.sidebar:
    stats = render_sidebar()
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
        "Context Chunks",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of document chunks to use as context"
    )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display evidence if available
        if message["role"] == "assistant" and "evidence" in message:
            with st.expander("📄 Evidence", expanded=False):
                for ev in message["evidence"]:
                    st.markdown(render_evidence_card(ev), unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about SEC filings..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        evidence_placeholder = st.empty()
        
        full_response = ""
        evidence_data = []
        
        # Show status while processing
        status_container = st.empty()
        with status_container:
            with st.status("Thinking...", expanded=True) as status:
                pass
        
        # Stream the response
        for chunk in ask_question(
            query=prompt,
            top_k=top_k,
            ticker=ticker_input if ticker_input else None,
            filing_type=filing_type if filing_type != "All" else None
        ):
            if chunk['type'] == 'content':
                full_response += chunk['data']
                response_placeholder.markdown(full_response + "▌")
            elif chunk['type'] == 'done':
                evidence_data = chunk['evidence']
                response_placeholder.markdown(full_response)
                status_container.empty()  # Remove status completely
                break
            elif chunk['type'] == 'error':
                response_placeholder.error(chunk['message'])
                status_container.empty()  # Remove status completely
                break
        
        # Display evidence
        if evidence_data:
            with evidence_placeholder.expander("📄 Evidence", expanded=False):
                for ev in evidence_data:
                    st.markdown(render_evidence_card(ev), unsafe_allow_html=True)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "evidence": evidence_data
        })

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()
