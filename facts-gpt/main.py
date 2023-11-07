from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def main():

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )

    loader = TextLoader(rf"C:\Users\adity\Desktop\langchain-dev\facts-gpt\facts.txt")

    docs = loader.load_and_split(text_splitter=text_splitter)

    db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory="emb"
    )


if __name__ == "__main__":
    main()
