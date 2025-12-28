import streamlit as st
import requests
import json

# ==============================
# Page config
# ==============================
st.set_page_config(
    page_title="Local Chatbot (Offline)",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Local Chatbot (Offline)")
st.caption("Powered by Ollama · Runs fully on your laptop")

# ==============================
# Sidebar controls
# ==============================
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        ["mistral", "llama3.1:8b-instruct"],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []

# ==============================
# Initialize chat state
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# Display chat history
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# User input
# ==============================
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Prepare request to Ollama
    payload = {
        "model": model_name,
        "messages": st.session_state.messages,
        "temperature": temperature,
        "stream": False
    }

    # Call Ollama API
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        assistant_reply = result["message"]["content"]

    except Exception as e:
        assistant_reply = f"❌ Error: {e}"

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
