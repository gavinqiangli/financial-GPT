from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

tool = YahooFinanceNewsTool()
tool.run("NVDA")

res = tool.run("AAPL")
print(res)