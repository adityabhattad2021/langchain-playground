from langchain.prompts import MessagesPlaceholder,HumanMessagePromptTemplate,ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory,FileChatMessageHistory,ConversationSummaryMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory(
    #     file_path="messages.json"
    # ),
    memory_key="messages",
    return_messages=True,
    llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["content","messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("(Press q to quit)>>: ")
    if user_input == "q":
        break
    print(chain({"content":user_input})["text"])

