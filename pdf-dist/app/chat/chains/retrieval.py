from langchain.chains import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain

class StreamingConversationalRetirevalChain(StreamableChain,ConversationalRetrievalChain):
    pass