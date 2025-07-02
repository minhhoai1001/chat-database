import os, re, json, sys
from langchain_core.messages import HumanMessage, SystemMessage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.mysql_client import MySQLClient
from tools.qdrant_client import QdrantVectorTool
from llm import BedrockLLM

def extract_tables(text):
    # Pattern to match CREATE TABLE statements and their documentation
    table_pattern = r"(CREATE TABLE.*?)(?=CREATE TABLE|\Z)"
    
    # Find all tables in the text
    tables = re.finditer(table_pattern, text, re.DOTALL)
    
    result = []
    for table in tables:
        table_text = table.group().strip()
        
        # Extract table name from CREATE TABLE statement
        table_name_pattern = r"CREATE TABLE\s+`?(\w+)`?\s*\("
        table_name_match = re.search(table_name_pattern, table_text, re.IGNORECASE)
        table_name = table_name_match.group(1) if table_name_match else ""
        
        # Extract DDL and documentation for each table
        dll_pattern = r"CREATE TABLE.*?(?=\/\*|$)"
        doc_pattern = r"\/\*.*?\*\/"
        
        dll_match = re.search(dll_pattern, table_text, re.DOTALL)
        doc_match = re.search(doc_pattern, table_text, re.DOTALL)
        
        table_dict = {
            "name": table_name,
            "dll": dll_match.group().strip() if dll_match else "",
            "sample": doc_match.group().strip() if doc_match else ""
        }
        result.append(table_dict)
    
    return result

if __name__ == "__main__":
    MYSQL_SERVER = os.environ['MYSQL_SERVER']
    MYSQL_PORT = os.environ['MYSQL_PORT']
    MYSQL_PASSWORD = os.environ['MYSQL_PASSWORD']
    
    qdrant_client = QdrantVectorTool(url="localhost", port=6333, collection_name="client_ddl")

    mysql_client = MySQLClient(MYSQL_SERVER, MYSQL_PORT, MYSQL_PASSWORD)
    schema = mysql_client.get_table_info()
    
    llm = BedrockLLM()
    # prompt = """
    # You are a data analyst assistant. Your task is to generate a short and clear description of the purpose of a database table.

    # You will receive 3 sample rows from a SQL table. Based only on this sample data, describe what this table is likely used for in 1-2 concise sentences. Focus on:
    # - What kind of information the table stores
    # - What its likely role is in a system
    # - Keywords that might be useful when selecting it for user questions

    # Avoid repeating values from the sample. Do not speculate beyond the data shown.

    # Format your answer as a single short paragraph, no more than 2 sentences.

    # Only output the description. Do not include the data in your answer.
    # """
    
    prompt = """
    You are a database expert helping summarize SQL table schemas for language models.

    Given the SQL DDL of a table, extract and write a **concise natural language description** in the following format:

    <lowercase_table_name>: A brief description of what the table stores. Mention the key columns such as timestamps, user IDs, types, or contents. If the table has foreign keys, describe what table and column it links to.

    Only include relevant and useful information. Avoid unnecessary SQL syntax or low-level implementation details.
    """
    
    # data = qdrant_client.search_embedding("Which is the most recent scan?")
    # for item in data:
    #     print(item.score)
    #     print(item.payload)
    #     print("-"*100)
    
    # print(schema)
    # tables = extract_tables(schema)
    # for table in tables:
    #     print("-"*100)
    #     dll = table["dll"]
    #     sample = table["sample"]
    #     name = table["name"]

    #     system = SystemMessage(content=prompt)
    #     human = HumanMessage(content=f"Here is the input SQL schema: {dll}")
    #     response = llm.invoke([system, human])
    #     print(response.content)
    #     qdrant_client.add_embedding(name, response.content, dll)
    names = ['projects', "scans"]
    for name in names:
        data = qdrant_client.get_dll_by_name(name)
        print(data)
        print("-"*100)