# import libraries
import os
from dotenv import load_dotenv
import constants as cts
from get_docs import GetDocuments as get_docs
from get_nearest_links import GetNearestLinks
import openai
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import tool
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
    content="You are atheist and your name is agent1, Strictly reply with your name and response and keep it in one sentence and act as agent1"
)
prompt1 = OpenAIFunctionsAgent.create_prompt(system_message=system_message1)

system_message2 = SystemMessage(
    content="You are religious and your name is agent2, Strictly reply with your name and response and keep it in one sentence and act as agent2"
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
    return len(word)


@tool
def get_info_on_philosophy(query: str) -> str:
    """Returns info relevant Philosophy"""
    # get top 3 results for query
    gnl = GetNearestLinks(query)
    top_links = gnl.get_links()

    # get docs from summaries json
    top_docs = get_docs(top_links).get_documents()
    # concatenate all summaries
    prompt_text = "\n".join([doc["summary"] for doc in top_docs])

    return prompt_text


tools = [get_info_on_philosophy]

agent1 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt1)
agent2 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt2)

agents = [agent1, agent2]


class GetResponse:
    def __init__(self, query: str = None):
        self.query = query
        self.prompt_text = None
        # get env variables
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # get openai gpt model to use
        self.gpt_model = cts.OPENAI_GPT_MODEL

    def get_response(self) -> str:
        # get prompt
        prompt_template = self.get_prompt()

        # get response from openai
        answer = self.get_openai_response(prompt_template)

        return answer

    def get_prompt(self) -> str:
        # get top 3 results for query
        gnl = GetNearestLinks(self.query)
        top_links = gnl.get_links()

        # get docs from summaries json
        top_docs = get_docs(top_links).get_documents()
        # concatenate all summaries
        self.prompt_text = "\n".join([doc["summary"] for doc in top_docs])

        # form prompt for openai
        prompt_template = f"""Use the below extract from articles on Philosophy to provide a summary in simple terms. Mould your summary to answer the subsequent question. 
        
        Start your response with "According to articles published by Stanford Encyclopedia of Philosphy". 
        
        If a summary cannot be provided, write "I don't know."

        Extract:
        \"\"\"
        {self.prompt_text}
        \"\"\"
        Question: {self.query}"""

        return prompt_template

    def get_openai_response(self, prompt_template: str) -> str:
        # set api key
        openai.api_key = self.openai_api_key
        # call openai
        openai_response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": "You can summarize texts on Philosophy."},
                {"role": "user", "content": prompt_template},
            ],
            model=self.gpt_model,
            temperature=0,
        )
        answer = openai_response["choices"][0]["message"]["content"]
        return answer


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
        if len(words) > 500:
            st.session_state.prompts = "".joint(words[-200:])

        with st.chat_message("ai", avatar="🦖"):
            message_placeholder = st.empty()
            full_response = ""
            agent_executor = AgentExecutor(agent=agent1, tools=tools, verbose=False)
            for response in agent_executor.run(st.session_state.prompts):
                # full_response += response.choices[0].delta.get("content", "")
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
                # full_response += response.choices[0].delta.get("content", "")
                full_response += response
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.prompts += str(" " + full_response)
        st.session_state.messages.append({"role": "agent2", "content": full_response})

        time.sleep(1)
