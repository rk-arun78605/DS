"""
Modern Streamlit UI for the chatbot - Integrated with FastAPI backend
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Analytics Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration
API_URL = "http://localhost:8000"

# ============================================================
# Custom CSS Styling
# ============================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px;
        padding: 14px 18px;
        margin: 12px 0;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f8f9fe 0%, #e8eaf6 100%);
        color: #333;
        border-radius: 18px;
        padding: 14px 18px;
        margin: 12px 0;
        max-width: 80%;
        border-left: 4px solid #667eea;
        word-wrap: break-word;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Data table */
    .data-table {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Metrics card */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 4px solid #ff9800;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 4px solid rgba(102, 126, 234, 0.1);
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State Management
# ============================================================

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None

# ============================================================
# Helper Functions
# ============================================================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def send_chat_message(question: str) -> Dict[str, Any]:
    """Send message to API and get response"""
    try:
        response = requests.post(
            f"{API_URL}/api/chat",
            json={
                "question": question,
                "chat_history": []
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "⏱️ Request timeout. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "🔌 Cannot connect to API. Is the server running?"}
    except Exception as e:
        return {"error": f"❌ Error: {str(e)}"}

def format_data_for_display(data: list) -> pd.DataFrame:
    """Format API response data for Streamlit"""
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)

# ============================================================
# Main UI Layout
# ============================================================

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #333; margin: 0;">🤖 AI Analytics Chatbot</h1>
        <p style="color: #666; font-size: 14px; margin-top: 8px;">Convert natural language to SQL insights</p>
    </div>
    """, unsafe_allow_html=True)

# API Status
col1, col2, col3 = st.columns([1, 2, 1])
with col3:
    st.session_state.api_connected = check_api_health()
    if st.session_state.api_connected:
        st.success("✅ API Connected")
    else:
        st.error("🔴 API Offline")

st.markdown("---")

# Chat area
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat history
for i, msg in enumerate(st.session_state.chat_history):
    if msg['role'] == 'user':
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
st.markdown("---")

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input(
        "Ask me anything about your data...",
        placeholder="e.g., Show top 5 overstock shops, Which items need replenishment?",
        key="chat_input"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True)

# Process message
if send_button and user_input:
    if not st.session_state.api_connected:
        st.error("❌ API is not connected. Please start the server: python chatbot_system/api.py")
    else:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response from API
        with st.spinner("🔍 Processing your query..."):
            response = send_chat_message(user_input)
        
        if "error" in response:
            st.error(response["error"])
            st.session_state.chat_history[-1] = {
                "role": "assistant",
                "content": response["error"]
            }
        else:
            # Add bot response to history
            bot_response = response.get("response", "No response")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": bot_response
            })
            
            # Display insights if available
            if response.get("insights"):
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(f"💡 **Insights:**\n{response['insights']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Display data if available
            if response.get("data") and response.get("row_count", 0) > 0:
                st.markdown(f"📊 **Results:** {response['row_count']} records found")
                
                df = format_data_for_display(response["data"])
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Display SQL used if available
            if response.get("sql_used"):
                with st.expander("🔍 View SQL Query"):
                    st.code(response["sql_used"], language="sql")
            
            # Rerun to update chat display
            st.rerun()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.subheader("📝 Sample Queries")
    sample_queries = [
        "Show me top 5 overstock items",
        "Which shops have low stock?",
        "What are the top selling departments?",
        "Analyze sales by item category",
        "Which items need replenishment?"
    ]
    
    for query in sample_queries:
        if st.button(query, key=f"query_{query}"):
            st.session_state.chat_input = query
            st.rerun()
    
    st.divider()
    
    st.subheader("🔧 Advanced")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📊 Quick Stats")
    if st.session_state.api_connected:
        try:
            health = requests.get(f"{API_URL}/health").json()
            st.metric("Database", "Connected ✅")
        except:
            st.metric("Database", "Offline ❌")
    
    st.divider()
    
    st.info("""
    **How to use:**
    1. Type your question naturally
    2. AI converts it to SQL
    3. View results and insights
    4. Download data as CSV
    """)
