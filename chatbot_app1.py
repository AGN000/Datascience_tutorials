import streamlit as st
import requests

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Lightweight Local Chatbot", page_icon="⚡")
st.title("⚡ Lightweight Local Chatbot")

# -----------------------------
# Prompt Example
# -----------------------------
with st.expander("📌 Prompt Example"):
    st.markdown("""
**System Prompt**
> You are a lightweight assistant that gives short, clear answers.

**User Prompt**
> What is machine learning?
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Model",
    ["phi3:mini", "tinyllama"]
)

system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a concise, helpful assistant."
)

# -----------------------------
# Session State
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# Reset chat if system prompt changes
if st.session_state.messages[0]["content"] != system_prompt:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# -----------------------------
# Display Messages
# -----------------------------
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# -----------------------------
# User Input
# -----------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------
    # Ollama Chat Request
    # -----------------------------
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model_name,
            "messages": st.session_state.messages,
            "stream": False
        }
    )

    assistant_reply = response.json()["message"]["content"]

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
