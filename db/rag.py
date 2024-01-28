import os
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_azure import AzureRag

# Set your Azure Text Analytics key and endpoint
azure_key = "YOUR_AZURE_TEXT_ANALYTICS_KEY"
azure_endpoint = "YOUR_AZURE_TEXT_ANALYTICS_ENDPOINT"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["AZURE_TEXT_ANALYTICS_KEY"] = azure_key
os.environ["AZURE_TEXT_ANALYTICS_ENDPOINT"] = azure_endpoint

# Load, chunk, and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits)

# Create an AzureRag instance
rag_chain = AzureRag(
    azure_key=azure_key,
    azure_endpoint=azure_endpoint,
    vectorstore=vectorstore,
    model_name="gpt-3.5-turbo",
    temperature=0,
)

# Retrieve and generate using the relevant snippets of the blog.
response = rag_chain.ask_question("What is Task Decomposition?")

# Access the response text
answer = response.get("answer", "")
print(answer)

# cleanup
vectorstore.delete_collection()