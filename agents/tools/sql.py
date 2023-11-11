import sqlite3
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List

conn = sqlite3.connect(rf"C:\Users\adity\Desktop\langchain-dev\agents\db.sqlite")

def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

def run_squlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occoured: {str(err)}"
    
def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'"+table+"'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name in ({tables});")
    return '\n'.join(row[0] for row in rows if row is not None)

class RunQueryArgsSchema(BaseModel):
    query:str

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Runs the given SQL query",
    func=run_squlite_query,
    args_schema=RunQueryArgsSchema
)

class DescribeTableArgsSchema(BaseModel):
    table_names:List[str]

describe_tables_tool = Tool.from_function(
    name="describle_tables",
    description="Returns the column names of the given tables",
    func=describe_tables,
    args_schema=DescribeTableArgsSchema
)


