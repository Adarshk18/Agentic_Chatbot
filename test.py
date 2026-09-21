from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage


config = {'configurable': {'thread_id': 'thread-1'}}
# response = chatbot.invoke({'messages': [HumanMessage(content="What are the benefit of learning Agentic AI in future?")]}, config=config)
# print(response['messages'][-1].content)

#now we are using stream
for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content="what are the benefits of working and learning with Agentic AI?")]},
    config=config,
    stream_mode='messages'
):

    if message_chunk.content:
        print(message_chunk.content, end='', flush=True)
