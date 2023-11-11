from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent,AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

from tools.sql import run_query_tool,list_tables,describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

load_dotenv()

tables = list_tables()
chat = ChatOpenAI(
    callbacks=[ChatModelStartHandler()]
)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database. Here is the list of all the availabe tables\n"
            f"The database has following tables:\n{tables}\n"
            "Do not make any assumptions about what tables exists or what columns exists\n"
            "Instead use the describe tables functions."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

tools = [
    run_query_tool,
    describe_tables_tool,
    write_report_tool
]

agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent = agent,
    verbose=True,
    tools=tools,
    memory=memory
)

agent_executor("how many users are there in the database")