import streamlit as st
import sqlite3
import re
from langchain_core.exceptions import OutputParserException

# Import agent_executor and extract_final_answer from your backend script
from app1 import agent_executor, extract_final_answer

# Streamlit page setup
st.set_page_config(page_title="E-commerce Chatbot", layout="wide")
st.title("🛒 E-commerce Assistant Chatbot")

# Chat history state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask something like 'check my order 3 status' or 'return order 5'")

if user_input:
    # Append user message
    st.session_state.chat_history.append(("user", user_input))

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        result = agent_executor.invoke({"input": user_input})
        raw_output = result.get("output", "") if isinstance(result, dict) else str(result)
        final_answer = extract_final_answer(raw_output)
    except OutputParserException as e:
        final_answer = f"[Parsing Error] {str(e)}"
    except Exception as e:
        final_answer = f"[Error] {str(e)}"

    # Append assistant response
    st.session_state.chat_history.append(("assistant", final_answer))

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(final_answer)

# Show previous chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
