from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st

## streamlit is a python package to build ui in python wiothout the use of html and css


st.title("Agentic chatbot with Langgraph")


config = {'configurable': {'thread_id': "3"}}
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])



user_input = st.chat_input('Ask anything...')
if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]},config=config)

    # ai_message = response['messages'][-1].content
    # st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    # with st.chat_message('assistant'):
    #     st.text(ai_message)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk ,metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})    



##  since suppose there is one state called 'messages' and after it moves from start till end then after refreshing or restar those state will be gone forever so to keep that state in db we use persistance 

