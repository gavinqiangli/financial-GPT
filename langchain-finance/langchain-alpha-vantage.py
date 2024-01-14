# Use the AlphaVantageAPIWrapper to get currency exchange rates.

import getpass
import os

os.environ["ALPHAVANTAGE_API_KEY"] = getpass.getpass()

from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper

alpha_vantage = AlphaVantageAPIWrapper()

alpha_vantage.run("USD", "JPY")