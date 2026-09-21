from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid


# Streamlit is a Python package to build UI in Python
# without the use of HTML and CSS.


def generate_thread_id():
    return str(uuid.uuid4())


def add_threads(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

    add_threads(st.session_state["thread_id"])


st.title("Agentic chatbot with Langgraph")


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


# ============================== Sidebar Threading Feature ==============================

st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask anything...")

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )

            if isinstance(message_chunk, AIMessage)
        )

        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })
