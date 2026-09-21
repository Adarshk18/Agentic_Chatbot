from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
import os 

load_dotenv()

llm = ChatOpenAI()


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}


checkpoint = MemorySaver()
graph = StateGraph(ChatState)

#add_nodes
graph.add_node('chat_node', chat_node)

#add_edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


chatbot = graph.compile(checkpointer=checkpoint)
# chatbot


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


    
    

