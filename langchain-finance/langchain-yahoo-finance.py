import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, initialize_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0.0)
tools = [YahooFinanceNewsTool()]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run(
    "What happens today with Microsoft stocks?",
)

agent_chain.run(
    "How does Microsoft feels today comparing with Nvidia?",
)