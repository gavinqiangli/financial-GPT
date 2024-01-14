import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper

os.environ["SERPAPI_API_KEY"] = os.getenv("SERP_API_KEY")
tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper())

res = tool.run("Tesla")
print(res)