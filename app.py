import streamlit as st
from openai import OpenAI
import requests
from urllib.request import urlopen
import certifi
import json
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

FMP_API_KEY = os.environ.get('FMP_API_KEY')

def get_jsonparsed_data(url):
    try:
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)

    except Exception as e:
        print(f"HTTP Error on url {url}: {e}")    
    

# Step 1: Retrieving Financial Statements
def get_financial_statements(ticker, limit, period, statement_type):
    if statement_type == "Income Statement":
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    elif statement_type == "Balance Sheet":
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    elif statement_type == "Cash Flow":
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    
    data = get_jsonparsed_data(url)
    print(data)

    return data


# Retrieving key financial metrics for given periods
def get_key_metrics(ticker, limit, period):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    data = get_jsonparsed_data(url)
    print(data)

    return data


# Step 2 / Option 1: Generate Financial Statements Summary with GPT-4
def generate_financial_statements_summary(financial_statements):
    """
    Generate a summary of financial statements for the statements using GPT-3.5 Turbo or GPT-4.
    """
    
    # Option 1: Create a summary of key financial metrics for all four periods
    summaries = []
    if financial_statements:
        for i in range(len(financial_statements)):
            
            summary = f"""
                For the period ending {financial_statements[i]['date']}, the company reported the following:
                {financial_statements[i]}
                """
            
            print(summary)
            summaries.append(summary)

    # Combine all summaries into a single string
    all_summaries = "\n\n".join(summaries)

    return all_summaries

   
# Step 2 / Option 2: Get Key Metrics directly from FMP API
def generate_key_metrics_summary(key_metrics):
    """
    Get a summary of financial key metrics using FMP API.
    Statement Analysis
    Key Metrics
    - Get key financial metrics for a company, including revenue, net income, earnings per share (EPS), and price-to-earnings ratio (P/E ratio). 
    - Assess a company's financial performance and compare it to its competitors.
    """

    # Option 2: Use FMP API to get key financial metrics for all four periods
    summaries = []
    if key_metrics:
        for i in range(len(key_metrics)):
            
            summary = f"""
                For the period ending {key_metrics[i]['date']}, the company reported the following:
                {key_metrics[i]}
                """
            
            print(summary)
            summaries.append(summary)

    # Combine all summaries into a single string
    all_summaries = "\n\n".join(summaries)

    return all_summaries


# Step 3: Call GPT-4 for final analysis
def final_analysis(all_summaries):

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trained to provide financial analysis based on financial statements.",
            },
            {
                "role": "user",
                "content": f"""
                Please analyze the following data and provide insights:\n{all_summaries}.\n 
                Write each section out as instructed in the summary section and then provide analysis of how it's changed over the time period.
                ...
                """
            }
        ]
    )

    return response.choices[0].message.content


# Step 4: Building the Streamlit App
# Selecting financial assistant 1: financial_statements
def financial_statements():
    st.title('GPT 4 Financial Statements')

    statement_type = st.selectbox("Select financial statement type:", ["Income Statement", "Balance Sheet", "Cash Flow"])

    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox("Select financial period:", ["Annual", "Quarter"]).lower()

    with col2:
        limit = st.number_input("Number of past financial statements to analyze:", min_value=1, max_value=10, value=4)
    
    ticker_input = st.text_input("Please enter the company ticker (such as MSFT):")
    company_name = st.text_input("Don't know company ticker? Try to enter company name instead (such as Microsoft):")
    if company_name:
        ticker_gpt = get_ticker_by_gpt(company_name)
        st.text(f"the company ticker for {company_name} is {ticker_gpt}")

    if st.button('Run'):
        with st.spinner("In progress..."):
            if ticker_input: # always prioritize to use ticker_input
                ticker = ticker_input
            else:
                ticker = ticker_gpt 
            if ticker:
                ticker = ticker.upper()
                financial_statements = get_financial_statements(ticker, limit, period, statement_type)

                with st.expander("View Financial Statements"):
                    st.dataframe(frame_data(financial_statements))

                statements_summary = generate_financial_statements_summary(financial_statements)

                final_statements_summary = final_analysis(statements_summary)

                st.write(f'Summary for {ticker} {statement_type}:\n\n {final_statements_summary}\n\n')
        st.success('Done!')

# Selecting financial assistant 2: financial_metrics
def financial_metrics():
    st.title('GPT 4 Financial Metrics')

    col1, col2 = st.columns(2)

    with col1:
        period = st.selectbox("Select financial period:", ["Annual", "Quarter"]).lower()

    with col2:
        limit = st.number_input("Number of past financial metrics to analyze:", min_value=1, max_value=10, value=4)
    
    ticker_input = st.text_input("Please enter the company ticker (such as MSFT):")
    company_name = st.text_input("Don't know company ticker? Try to enter company name instead (such as Microsoft):")
    if company_name:
        ticker_gpt = get_ticker_by_gpt(company_name)
        st.text(f"the company ticker for {company_name} is {ticker_gpt}")

    if st.button('Run'):
        with st.spinner("In progress..."):
            if ticker_input: # always prioritize to use ticker_input
                ticker = ticker_input
            else:
                ticker = ticker_gpt    
            if ticker:
                ticker = ticker.upper()
                key_metrics = get_key_metrics(ticker, limit, period)
            
                with st.expander("View Key Metrics"):
                    st.dataframe(frame_data(key_metrics))

                key_metrics_summary = generate_key_metrics_summary(key_metrics)

                final_key_metrics_summary = final_analysis(key_metrics_summary)

                st.write(f'Summary for {ticker} key metrics:\n\n {final_key_metrics_summary}\n\n')
        st.success('Done!')


# Panda frame the data
def frame_data(data):
    if isinstance(data, list) and data:
        return pd.DataFrame(data)
    else:
        st.error("Unable to fetch financial statements. Please ensure the ticker is correct and try again.")
        return pd.DataFrame()


# get company ticker by name
# however, yahoo finance api only accepts user-agent by browser request, not by streamlit app
def get_ticker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    print(res.status_code)
    print(res.text)

    data = res.json()

    company_code = data['quotes'][0]['symbol']

    if company_code:
        print(f"The ticker symbol for {company_name} is {company_code}")
    else:
        print(f"Could not find the ticker symbol for {company_name}")

    return company_code

# use GPT to get ticker name, works perfectly
def get_ticker_by_gpt(company_name):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an Financial expert that knows Stock Ticker Symbol for all listed companies",
            },
            {
                "role": "user",
                "content": f"""
                what is Stock Ticker Symbol for {company_name}.
                only provide the ticker name as response, nothing else. 
                """
            }
        ]
    )

    company_code = response.choices[0].message.content

    if company_code:
        print(f"The ticker symbol for {company_name} is {company_code}")
    else:
        print(f"Could not find the ticker symbol for {company_name}")

    return company_code
        

# finally, to run our application
def main():
    st.set_page_config(page_title="AI Financial Analyst", page_icon=":bird:")

    st.sidebar.title('AI Financial Analyst :bird:')
    st.sidebar.markdown("Demo by [Qiang Li](https://www.linkedin.com/in/qianglil/). All rights reserved.")

    # this markdown is for hiding "github" button
    st.markdown("<style>#MainMenu{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.markdown("<style>footer{visibility: hidden;}</style>", unsafe_allow_html=True)
    st.markdown("<style>header{visibility: hidden;}</style>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{display: none;} 
    </style>
    """,
    unsafe_allow_html=True
    )

    st.sidebar.header("Settings")
    app_mode = st.sidebar.selectbox("Choose your AI assistant:",
        ["Financial Statements", "Financial Metrics"])
    if app_mode == 'Financial Statements':
        financial_statements()
    if app_mode == 'Financial Metrics':
        financial_metrics()    


if __name__ == '__main__':
    main()