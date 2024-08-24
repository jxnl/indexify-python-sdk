import dotenv
import requests
from langchain_community.document_loaders import TextLoader

from langchain.text_splitter import CharacterTextSplitter

# specify your OPENAI_API_KEY in .env file
dotenv.load_dotenv()


# load state of the union text into langchain TextLoader
url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)

with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()


# chunk text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)


# initialize indexify client
from indexify.client import Document, IndexifyClient

client = IndexifyClient()


# Bind openai embedding extractor
client.add_extraction_policy(
    "openai-embedding-ada-002-extractor",
    "openai",
    labels_eq="source:state_of_the_union",
)


# Add Documents to repository
import time

docs = [Document(doc.page_content, {"source": "state_of_the_union"}) for doc in chunks]
client.add_documents(docs)
time.sleep(10)

# Setup indexify langchain retriever
from indexify_langchain import IndexifyRetriever

params = {"name": "openai.embedding", "top_k": 9}
retriever = IndexifyRetriever(client=client, params=params)

# Setup Chat Prompt Template
from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


# Ask llm question with retriever context
from langchain_openai import ChatOpenAI

from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


query = "Who are the speakers?"
print(rag_chain.invoke(query))
