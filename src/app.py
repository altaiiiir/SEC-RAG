import streamlit as st
import requests
import json
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SEC EDGAR Chat",
    page_icon="💬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .evidence-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #0066cc;
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
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'stop_generation' not in st.session_state:
    st.session_state.stop_generation = False

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display evidence if available
        if message["role"] == "assistant" and "evidence" in message:
            with st.expander("📄 Evidence", expanded=False):
                for i, ev in enumerate(message["evidence"], 1):
                    st.markdown(f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <div>
                                <strong>{ev['ticker']}</strong> | {ev['filing_type']} | {ev['filing_date'] or 'N/A'}
                            </div>
                            <span class="similarity-badge">{ev['similarity']*100:.0f}% match</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #333;">
                            {ev['content'][:300]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Generate response if we just added a user message
if st.session_state.is_generating and len(st.session_state.chat_history) > 0:
    last_message = st.session_state.chat_history[-1]
    if last_message["role"] == "user":
        user_query = last_message["content"]
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            evidence_placeholder = st.empty()
            
            full_response = ""
            evidence_data = []
            
            # Stream the response
            for chunk in ask_question(
                query=user_query,
                top_k=top_k,
                ticker=ticker_input if ticker_input else None,
                filing_type=filing_type if filing_type != "All" else None
            ):
                # Check if user stopped generation
                if st.session_state.stop_generation:
                    response_placeholder.markdown(full_response + "\n\n*[Response stopped by user]*")
                    st.session_state.is_generating = False
                    st.session_state.stop_generation = False
                    break
                
                if chunk['type'] == 'content':
                    full_response += chunk['data']
                    response_placeholder.markdown(full_response + "▌")
                elif chunk['type'] == 'done':
                    evidence_data = chunk['evidence']
                    response_placeholder.markdown(full_response)
                    st.session_state.is_generating = False
                    break
                elif chunk['type'] == 'error':
                    response_placeholder.error(chunk['message'])
                    st.session_state.is_generating = False
                    break
            
            # Display evidence
            if evidence_data:
                with evidence_placeholder.expander("📄 Evidence", expanded=False):
                    for i, ev in enumerate(evidence_data, 1):
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <div>
                                    <strong>{ev['ticker']}</strong> | {ev['filing_type']} | {ev['filing_date'] or 'N/A'}
                                </div>
                                <span class="similarity-badge">{ev['similarity']*100:.0f}% match</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #333;">
                                {ev['content'][:300]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add assistant response to chat history if not stopped
            if not st.session_state.stop_generation and full_response:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "evidence": evidence_data
                })
        
        # Reset generation state and rerun to show input again
        if not st.session_state.is_generating:
            st.rerun()

# Chat input with custom send/stop button
col1, col2 = st.columns([6, 1])
with col1:
    prompt = st.text_input(
        "Message",
        key="chat_input",
        placeholder="Ask a question about SEC filings...",
        label_visibility="collapsed",
        disabled=st.session_state.is_generating
    )
with col2:
    if st.session_state.is_generating:
        if st.button("⏸️ Stop", use_container_width=True, type="secondary"):
            st.session_state.stop_generation = True
    else:
        send_button = st.button("Send", use_container_width=True, type="primary")

# Process message when send is clicked
if not st.session_state.is_generating and prompt and ('send_button' in locals() and send_button):
    st.session_state.is_generating = True
    st.session_state.stop_generation = False
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.rerun()

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()
