from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


import requests
import math
import os
import sqlite3
import tempfile

load_dotenv()


llm = ChatOpenAI()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def ingest_rag_document(uploaded_file):
    DB_PATH = "faiss_db"

    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        file_path = temp_file.name

    try:
        # -----------------------------
        # 1. Select loader
        # -----------------------------
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)

        elif file_extension == ".txt":
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        elif file_extension == ".md":
            loader = TextLoader(
                file_path,
                encoding="utf-8"
            )

        elif file_extension == ".csv":
            loader = CSVLoader(file_path)

        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)

        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}"
            )

        # -----------------------------
        # 2. Load document
        # -----------------------------
        docs = loader.load()

        print(f"📄 File: {file_name}")
        print(f"📄 Documents loaded: {len(docs)}")

        if not docs:
            raise ValueError(
                f"No content could be extracted from {file_name}"
            )

        # -----------------------------
        # 3. Add metadata
        # -----------------------------
        for doc in docs:
            doc.metadata["source"] = file_name

        # -----------------------------
        # 4. Split into chunks
        # -----------------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250
        )

        chunks = splitter.split_documents(docs)

        print(f"🧩 Chunks created: {len(chunks)}")

        if not chunks:
            raise ValueError(
                f"No text chunks were created from {file_name}. "
                "The PDF may contain scanned/image-only pages or no extractable text."
            )

        # Check actual text
        non_empty_chunks = [
            chunk for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        print(f"📝 Non-empty chunks: {len(non_empty_chunks)}")

        if not non_empty_chunks:
            raise ValueError(
                f"No readable text was extracted from {file_name}. "
                "If this is a scanned PDF, OCR is required."
            )

        chunks = non_empty_chunks

        # -----------------------------
        # 5. Create / update FAISS
        # -----------------------------
        index_file = os.path.join(
            DB_PATH,
            "index.faiss"
        )

        pkl_file = os.path.join(
            DB_PATH,
            "index.pkl"
        )

        faiss_exists = (
            os.path.exists(index_file)
            and os.path.exists(pkl_file)
        )

        if faiss_exists:

            print("🔄 Existing FAISS DB found")

            vector_store = FAISS.load_local(
                DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            vector_store.add_documents(chunks)

        else:

            print("🆕 Creating new FAISS DB")

            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

        # -----------------------------
        # 6. Save
        # -----------------------------
        os.makedirs(DB_PATH, exist_ok=True)

        vector_store.save_local(DB_PATH)

        print("✅ FAISS DB saved")

        return {
            "success": True,
            "filename": file_name,
            "chunks": len(chunks)
        }

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the uploaded documents.

    Use this tool when the user asks factual or conceptual
    questions that may be answered using the uploaded documents.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the uploaded documents."
    
    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """Useful for simple math calculations"""
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtin__": {}}, allowed)
        return str(result)

    except Exception as e:
        print(f"Error in calculation: {str(e)}")


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=DHUVDLZVG9P6TN1H"
    r = requests.get(url)
    return r.json()


# Make tool list
tools = [get_stock_price, search_tool, calculator, rag_tool]

# make the llm tool aware
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    system_message = SystemMessage(
    content=
        "You are a helpful Agentic Chatbot with access to several tools.\n"

        "Tool usage instructions:\n"
        "- Use 'rag_tool' for questions about the uploaded documents.\n"
        "- Always retrieve relevant document content before answering "
        "document-related questions.\n"
        "- Use 'search_tool' for current events, recent information, "
        "or information that requires an internet search.\n"
        "- Use 'calculator' for mathematical calculations.\n"
        "- Use 'get_stock_price' when the user asks for the current "
        "price of a stock.\n"

        "Answer general questions directly when no tool is required.\n"
        "Do not invent information from the uploaded documents.\n"
        "If the user asks about an uploaded document but no document "
        "is available, ask them to upload one.\n"
        "After receiving a tool result, provide a clear and helpful "
        "final answer."
)
    messages = [system_message, *state['messages']]
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
# add_edges

graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpoint)
# chatbot


# Create table for chat titles
conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
""")

conn.commit()


def save_chat_title(thread_id, title):

    conn.execute(
        """
        INSERT OR REPLACE INTO chat_titles (thread_id, title)
        VALUES (?, ?)
        """,
        (thread_id, title)
    )

    conn.commit()


def get_all_threads():

    all_threads = {}

    # Get all existing LangGraph threads
    for checkpoints in checkpoint.list(None):

        thread_id = checkpoints.config[
            "configurable"
        ]["thread_id"]

        if thread_id not in all_threads:
            all_threads[thread_id] = "New Chat"

    # Get saved titles
    cursor = conn.execute(
        "SELECT thread_id, title FROM chat_titles"
    )

    for thread_id, title in cursor.fetchall():

        if thread_id in all_threads:
            all_threads[thread_id] = title

    return all_threads

# CONFIG = {"configurable": {"thread_id": "default_thread"}}

# res = chatbot.invoke(
#     {"messages": [HumanMessage(content="Hello, how are you?")]}
#     , config=CONFIG
# )
# print(res)

# thread_id = "1"

# initial_state = {
#     'messages': [HumanMessage(content='what is Inference engineering in 3 points? Is it related to optimizing tokens, for GPUs')]
# }
# config = {'configurable': {'thread_id': thread_id}}
# response = chatbot.invoke(initial_state,config=config)
# print(response['messages'][-1].content)


# while True:
#     user_message = input('Type here: ')

#     print('User: ', user_message)

#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break
#     config = {'configurable': {'thread_id': "1"}}
#     response = chatbot.invoke({'messages': [HumanMessage(content=user_message)]},config=config)
#     print('AI: ', response['messages'][-1].content)
