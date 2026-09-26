from agentic_chatbot_rag_backend import (
    chatbot,
    ingest_rag_document,
    get_all_threads,
    save_chat_title
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
import streamlit as st
import uuid
import traceback

# Streamlit is a Python package to build UI in Python
# without the use of HTML and CSS.

llm = ChatOpenAI(model="gpt-4o-mini")


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

    # Reset URL so refresh doesn't reload the old thread
    st.query_params["thread_id"] = st.session_state["thread_id"]


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
    qp_thread = st.query_params.get("thread_id")

    if qp_thread:
        st.session_state["thread_id"] = qp_thread
        messages = load_conversation(qp_thread)

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
    else:
        st.session_state["thread_id"] = generate_thread_id()

st.query_params["thread_id"] = st.session_state["thread_id"]


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "uploaded_file_keys" not in st.session_state:
    st.session_state["uploaded_file_keys"] = set()

if "failed_file_keys" not in st.session_state:
    st.session_state["failed_file_keys"] = set()


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

user_input = st.chat_input(
    "Ask anything...",
    accept_file="multiple",
    file_type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "csv",
        "pptx"
    ]
)


if user_input:

    # ============================== Get Input ==============================

    text = user_input.text
    uploaded_files = user_input.files
    document_processing_failed = False

    # ============================== Process Documents ==============================

    for uploaded_file in uploaded_files:

        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        # Skip files already successfully processed
        if file_key in st.session_state["uploaded_file_keys"]:
            continue

        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            try:

                result = ingest_rag_document(
                    uploaded_file
                )

                # Remember processed file
                st.session_state[
                    "uploaded_file_keys"
                ].add(file_key)

                st.session_state["failed_file_keys"].discard(file_key)

                st.toast(
                    f"✅ {uploaded_file.name} processed!"
                )

            except Exception as e:

                st.session_state["failed_file_keys"].add(file_key)

                st.warning(
                    f"⚠️ Couldn't process "
                    f"{uploaded_file.name}: {e} — continuing without it."
                )
    # ============================== No Text ==============================

    # If user only uploaded a document and didn't ask anything,
    # don't send an empty message to LangGraph.

    if not text:

        st.stop()


    # ============================== Build LLM-facing message ==============================

    if uploaded_files:
        file_list = ", ".join(f.name for f in uploaded_files)

        llm_text = (
            f"{text}\n\n"
            f"[System note: The user just attached the following file(s) in this message: "
            f"{file_list}. They have already been processed and indexed. "
            f"Use the rag_tool to retrieve relevant content from them before answering.]"
        )
    else:
        llm_text = text

    # ============================== Create Chat Title ==============================
    

    thread_id = st.session_state["thread_id"]

    if thread_id not in st.session_state["chat_threads"]:

        title = generate_chat_title(text)

        add_threads(
            thread_id,
            title
        )

        save_chat_title(
            thread_id,
            title
        )

    # ============================== Add User Message ==============================

    st.session_state["message_history"].append({
        "role": "user",
        "content": text
    })

    with st.chat_message("user"):

        st.write(text)

        # Display attached files
        for uploaded_file in uploaded_files:

            st.caption(
                f"📎 {uploaded_file.name}"
            )

    # ============================== LangGraph Config ==============================

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # ============================== AI Response ==============================

    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=llm_text
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # ==============================
                # Tool Status
                # ==============================

                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using {tool_name} ...",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using {tool_name} ...",
                            state="running",
                            expanded=True,
                        )

                # ==============================
                # Stream AI Tokens
                # ==============================

                if isinstance(
                    message_chunk,
                    AIMessage
                ):

                    yield message_chunk.content

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # ==============================
        # Finish Tool Status
        # ==============================

        if status_holder["box"] is not None:

            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False
            )

        # ==============================
        # Save AI Message
        # ==============================

        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })
