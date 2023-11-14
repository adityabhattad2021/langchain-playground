from app.chat.chains.retrieval import StreamingConversationalRetirevalChain
from app.chat.models import ChatArgs
from app.chat.vector_stores.pipecone import build_retriever
from app.chat.llms.chatopenai import build_llm
from app.chat.memories.sql_memory import build_memory
from langchain.chat_models import ChatOpenAI



def build_chat(chat_args: ChatArgs):
    retriever = build_retriever(chat_args=chat_args)
    llm = build_llm(chat_args=chat_args)
    memory = build_memory(chat_args=chat_args)

    return StreamingConversationalRetirevalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        condense_question_llm=ChatOpenAI(streaming=False),
    )
