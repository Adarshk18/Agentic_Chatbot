from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os 
import sqlite3

load_dotenv()

llm = ChatOpenAI()


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

conn  = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

#add_nodes
graph.add_node('chat_node', chat_node)

#add_edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


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


    
    

