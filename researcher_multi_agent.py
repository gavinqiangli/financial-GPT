import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from database import connect_2_db
from pymongo import MongoClient

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# Load Pinecone API key
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
# Set Pinecone environment. Find next to API key in console
pinecone_env = os.environ.get('PINECONE_ENVIRONMENT')
pinecone_index = os.environ.get('PINECONE_INDEX_NAME')
pinecone_namespace = os.environ.get('PINECONE_NAME_SPACE')

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# 1. Tool for RAG retrieval from knowledge base
def knowledgebase(objective):

    # chat completion llm
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
       
    # pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    embeddings = OpenAIEmbeddings()

    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(pinecone_index)
    
    vectorstore = Pinecone.from_existing_index (
        index_name=pinecone_index, embedding=embeddings, namespace=pinecone_namespace
    )

    # output = vectorstore.similarity_search(
    #     objective,  # our search query
    #     k=3  # return 3 most relevant docs
    # )
    # print(output)

    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    output = qa.invoke(objective)
    print(output)
    return output

# knowledgebase("what is revenue forecast of China Telecom?")

# 2. Tool for search
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# 3. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {"url": url}

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()

        # in case the url is a pdf document to scrape...
        # this is too costly to run with large pdf...
        # "This model's maximum context length is 16385 tokens. 
        # However, your messages resulted in 74343 tokens. Please reduce the length of the messages."
        if url.endswith(".pdf"):
            # check if this url has been processed with RAG before
            _, rag_url_history = connect_2_db()
            find_url = rag_url_history.find_one({"url": url})
            # if it is new url
            if find_url is None:
                # begin to RAG process
                content = scrape_pdf_with_pymupdf(url)
                text = rag_ingestion_retrieval(objective, content)
                # store new url to db        
                rag_url_history.insert_one({"url": url})
                print("new pdf url saved to db!")


        print("CONTENTTTTTT:", text)

        if len(text) > 10000:
            # option A:
            output = summary(objective, text)
            # option B:
            # output = map_reduce(objective, text)
            # option C:
            # output = rag_retrieval(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def scrape_pdf_with_pymupdf(url) -> str:
        """Scrape a pdf with pymupdf

        Args:
            url (str): The url of the pdf to scrape

        Returns:
            str: The text scraped from the pdf
        """
        loader = PyMuPDFLoader(url)
        doc = loader.load()
        return doc
        #return str(doc)


# Tool for scraping and summarization
# Three options: A, B, C
# option A: smaller chunk size that can fit 16k token context window
# option B: larger chunk size that needs to map reduce to 16k token
# option C: extra larger chunk size that could be turn into embedding for RAG retrieval

# option C: turn into embedding for RAG retrieval
def rag_ingestion_retrieval(objective, content):
       
    # 1. Vectorise the data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=200
    )
    # docs = text_splitter.create_documents([content])
    split_docs = text_splitter.split_documents(content)
  
    print('creating vector store...')

    # pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    embeddings = OpenAIEmbeddings()
   
    vectorstore = Pinecone.from_documents (
        split_docs, embedding=embeddings, index_name=pinecone_index, namespace=pinecone_namespace
    )

    index = pinecone.Index(pinecone_index)

    print("Index created!")

    # 2. Function for similarity search
    # output = vectorstore.similarity_search(
    #     objective,  # our search query
    #     k=3  # return 3 most relevant docs
    # )
    # print(output)
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    output = qa_with_sources.invoke(objective)
    print(output)
    return output


   

# option B: use map reduce chain
def map_reduce(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    split_docs = text_splitter.split_documents(docs)
    # "This model's maximum context length is 16385 tokens. 
    # However, your messages resulted in 74343 tokens. Please reduce the length of the messages."

    # Map
    map_prompt_template = """
    The following is a set of documents
    {split_docs}
    Based on this list of docs, please consolidate summary for {objective}.
    Helpful Answer:
    """
    map_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["split_docs", "objective"]
    )
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_prompt_template = """The following is set of summaries:
    {summaries}
    Take these and distill it into a final consolidated summary for {objective}. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate(
        template=reduce_prompt_template, input_variables=["summaries", "objective"]
    )

    # Run chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="summaries"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=16000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="split_docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    output = map_reduce_chain.run(input_documents=split_docs, objective=objective)

    return output


# option A: use summary chain
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name='knowledgebase',
        func=knowledgebase,
        description=(
            'always use this tool first when answering general knowledge queries to get '
            'more information about the topic'
        )
    ),
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class financial researcher, who can do detailed research on any financial topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should only focus on financial related research and ignore irrelevant web search
            2/ You should do enough research to gather as much information as possible about the objective
            3/ At first, you should always try to retrive relevant information from knowledge base to get possible answers
            4/ After you have got relevant answers from knowledge base, you should finish the research process and return the results
            5/ If you really can't get relevant answers from knowledge base, you can start to search web for public information
            6/ If there are url of relevant links & articles, you will scrape it to gather more information
            7/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            8/ You should not make things up, you should only write facts & data that you have gathered
            9/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            10/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


def generate_premium_response(prompt_input):
    content = agent({"input": prompt_input})
    actual_content = content["output"]
    return actual_content

# generate_premium_response("provide detailed ESG analysis of China Telecom")
