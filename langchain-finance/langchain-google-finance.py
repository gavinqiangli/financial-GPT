import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
llm = ChatOpenAI(temperature=0.0)
tools = load_tools(["google-scholar", "google-finance"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("what is google's stock")