# import libraries
import os
from dotenv import load_dotenv
import constants as cts
from get_docs import GetDocuments as get_docs
from get_nearest_links import GetNearestLinks
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import tool
from langchain.agents import AgentExecutor

import streamlit as st


llm = ChatOpenAI(temperature=0, openai_api_key='')

system_message1 = SystemMessage(content="You are atheist and your name is agent1 and you in debate with agent2 and user, keep conversation sort")
prompt1 = OpenAIFunctionsAgent.create_prompt(system_message=system_message1)

system_message2 = SystemMessage(content="You are religious and your name is agent2 and you in debate with agent1 and user, keep conversation sort")
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
def get_info_on_philosophy(query:str) -> str:
    """Returns info relevant Philosophy"""
        # get top 3 results for query        
    gnl = GetNearestLinks(query)
    top_links = gnl.get_links()
                
        # get docs from summaries json
    top_docs = get_docs(top_links).get_documents()
        # concatenate all summaries
    prompt_text = '\n'.join([doc['summary'] for doc in top_docs])
        
        # # form prompt for openai
        # prompt_template = f"""Use the below extract from articles on Philosophy to provide a summary in simple terms. Mould your summary to answer the subsequent question. 
        
        # Start your response with "According to articles published by Stanford Encyclopedia of Philosphy". 
        
        # If a summary cannot be provided, write "I don't know."

        # Extract:
        # \"\"\"
        # {self.prompt_text}
        # \"\"\"
        # Question: {self.query}"""

    return prompt_text
    

tools = [tool1, tool2, get_info_on_philosophy]

agent1 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt1)
agent2 = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt2)

agents = [agent1, agent2]



# agent_executor = AgentExecutor(agent=agents, verbose=True)

# agent_executor.run("how many letters in the word educa?")

class GetResponse:
    def __init__(self, query: str=None):
        self.query = query
        self.prompt_text = None
        # get env variables
        load_dotenv()
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
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
        self.prompt_text = '\n'.join([doc['summary'] for doc in top_docs])
        
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
                {'role': 'system', 'content': 'You can summarize texts on Philosophy.'},
                {'role': 'user', 'content': prompt_template},
            ],
            model=self.gpt_model,
            temperature=0,
        )
        answer = openai_response['choices'][0]['message']['content']        
        return answer
    
if __name__ == "__main__":
    # user_query = input("Hello, Welcome to ThinkAI. This is Nomí, ask me a question: ")
    print("Hello welcome to debate!")
    prompt = ''
    user_query = input()
    prompt+= str(' user: '+user_query)
    while True:
        agent_executor = AgentExecutor(agent=agent1, tools=tools, verbose=False)
        reponse1 = agent_executor.run(prompt)
        prompt+= str(reponse1)
        print(reponse1)
        agent_executor = AgentExecutor(agent=agent2, tools=tools, verbose=False)
        reponse2 = agent_executor.run(prompt)
        prompt+= str(reponse2)
        print(reponse2)
        # user_query = input()
        # prompt+= str(' user: '+user_query)
        if user_query == "":
            break
    
    

user = st.chat_input("Say something")
if user:
    st.write(f"User has sent the following prompt: {user}")
    # gr = GetResponse(user_query)
    # agent_executor.run("how many letters in the word educa?")
    # print(gr.get_response())