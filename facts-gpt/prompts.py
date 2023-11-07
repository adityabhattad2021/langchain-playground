from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever
from dotenv import load_dotenv
import langchain

langchain.debug = True

load_dotenv()

embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory=rf"C:\Users\adity\Desktop\langchain-dev\facts-gpt\emb",
    embedding_function=embeddings
)

chat = ChatOpenAI()

retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    vectorDB=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff"
)

result = chain.run("Tell me a fact about snail")

print(result)



