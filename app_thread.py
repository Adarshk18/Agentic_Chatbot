from agentic_chatbot_backend import (
    chatbot,
    get_all_threads,
    save_chat_title
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import uuid


# Streamlit is a Python package to build UI in Python
# without the use of HTML and CSS.

llm = ChatOpenAI()


def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(message):
    response = llm.invoke(
        f"""
        Generate a short title for this conversation.

        User message: {message}

        Rules:
        - Maximum 5 words
        - No quotes
        - Keep it descriptive
        - Return only the title
        """
    )

    return response.content.strip()


def add_threads(thread_id, title):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title


def reset_chat():
    # Create a new thread
    st.session_state["thread_id"] = generate_thread_id()

    # Clear current messages
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


st.title("Agentic chatbot with Langgraph")


# ============================== Session State ==============================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# ============================== Sidebar ==============================

st.sidebar.title("My Conversations")


# Create a new chat
if st.sidebar.button("New Chat"):

    reset_chat()

    # Rerun Streamlit app
    st.rerun()


# Display all conversation threads in reverse order
# Newest conversation appears first

for thread_id, title in reversed(
    list(st.session_state["chat_threads"].items())
):

    if st.sidebar.button(
        title,
        key=thread_id
    ):

        # Set selected thread as current thread
        st.session_state["thread_id"] = thread_id

        # Load messages from LangGraph
        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:

            if isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages

        st.rerun()


# ============================== Display Chat History ==============================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ============================== User Input ==============================

user_input = st.chat_input("Ask anything...")


if user_input:

    thread_id = st.session_state["thread_id"]

    # ============================== Create Chat Title ==============================

    # Generate title only when this is the first message
    # of a new conversation.

    if thread_id not in st.session_state["chat_threads"]:

        title = generate_chat_title(user_input)

        add_threads(
            thread_id,
            title
        )

        # Save title permanently in SQLite
        save_chat_title(
            thread_id,
            title
        )

    # ============================== Add User Message ==============================

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    # ============================== LangGraph Config ==============================


    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
            
    }

    # ============================== AI Response ==============================

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

    # ============================== Save AI Message ==============================

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
