from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

import requests
import math
import os 
import sqlite3

load_dotenv()

llm = ChatOpenAI()

# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression:str)-> str:
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
def get_stock_price(symbol: str)-> dict:
    """Get the current stock price for a symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=DHUVDLZVG9P6TN1H"
    r = requests.get(url)
    return r.json()


#Make tool list
tools = [get_stock_price, search_tool, calculator]

#make the llm tool aware 
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}



tool_node = ToolNode(tools)

conn  = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node", tools_condition)
#add_edges

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


    
    

