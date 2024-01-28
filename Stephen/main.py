import os
import json

from dotenv import load_dotenv

load_dotenv()

print("It starts now")

from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS



history = ChatMessageHistory()


llm = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    deployment_name="Test03",
    model_name="gpt-3.5-turbo"
)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant at the twelth grade level."),
    ("user", "{input}")
])

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template = template
)

memory = ConversationBufferMemory (memory_key = "chat_history", return_messages=True)

conversation = ConversationChain(
    llm = llm, verbose = 0, memory = ConversationBufferMemory()
)

conversation_with_summary = ConversationChain(
    llm=llm,
    # We set a very low max_token_limit for the purposes of testing.
    memory=ConversationBufferMemory(llm=llm),
    verbose=True,
)
#print(conversation_with_summary.predict(input="Hi, what's up?"))

def Roughbot(prompt):
    response = conversation.predict(input=prompt)
    return response

'''
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye", "end"]:
            break

        response = Roughbot(user_input)
        print("Chatbot: ", response)
'''
print("embedding")
embeddings = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1)
loader = DirectoryLoader('/workspaces/hoya_hacks/Stephen/HoyaDocs/UMD', glob="*.txt", loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})

documents = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


db = FAISS.from_documents(documents=docs, embedding=embeddings)

#Test

# Adapt if needed
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                           retriever=db.as_retriever(),
                                           condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                           return_source_documents=True,
                                           verbose=True)

chat_history = []
query = "Where can current undergrad UMD students find how much tuitions costs?"
result = qa({"question": query, "chat_history": chat_history})

print("Question:", query)
print("Answer:", result["answer"])
