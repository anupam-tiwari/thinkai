# import libraries
import os
from dotenv import load_dotenv
import constants as cts
from get_docs import GetDocuments as get_docs
from get_relevant_text import GetRelevantText
import openai
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.agents import AgentExecutor
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
import time

st.title("LLM friends")

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=openai_key,
)

system_message1 = SystemMessage(
    content="You are atheist and your name is agent1, Strictly reply with your name and response and keep it in one sentence and act as agent1, use get_info_on_philosophy() function to answer and keep on debating"
)
prompt1 = OpenAIFunctionsAgent.create_prompt(system_message=system_message1)

system_message2 = SystemMessage(
    content="You are religious and your name is agent2, Strictly reply with your name and response and keep it in one sentence and act as agent2, use get_info_on_philosophy() function to answer and keep on debating"
)
prompt2 = OpenAIFunctionsAgent.create_prompt(system_message=system_message2)


@tool
def tool1() -> int:
    """returns 1"""
    return 1


@tool
def tool2() -> int:
    """returns 2"""
    return 2


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word) +1


@tool
def get_info_on_philosophy(query: str) -> str:
    """Returns info relevant Philosophy"""
    grt = GetRelevantText(query)
    prompt_text = grt.get_relevant_text()

    return prompt_text


tools = [
    Tool(
            name = "Search",
            func=get_info_on_philosophy,
            description="use this tool when you need to get info on Philosophy"
        ),
]

agent1 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt1)
agent2 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt2)

agents = [agent1, agent2]

st.session_state.setdefault("prompts", "")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.prompts += str(" user: " + prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
    while True:
        words = st.session_state.prompts.split()
        if len(words) > 200:
            st.session_state.prompts = "".join(words[-200:])

        with st.chat_message("ai", avatar="🦖"):
            message_placeholder = st.empty()
            full_response = ""
            agent_executor = AgentExecutor(agent=agent1, tools=tools, verbose=False)
            for response in agent_executor.run(st.session_state.prompts):
                full_response += response
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.prompts += str(" " + full_response)
        st.session_state.messages.append({"role": "agent1", "content": full_response})

        time.sleep(1)

        with st.chat_message("ai", avatar="🧑‍💻"):
            message_placeholder = st.empty()
            full_response = ""
            agent_executor = AgentExecutor(agent=agent2, tools=tools, verbose=False)
            for response in agent_executor.run(st.session_state.prompts):
                full_response += response
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.prompts += str(" " + full_response)
        st.session_state.messages.append({"role": "agent2", "content": full_response})

        time.sleep(1)
        
        
