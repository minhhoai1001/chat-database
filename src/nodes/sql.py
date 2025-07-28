import os, json, datetime
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import GraphState
from src.llm import BedrockLLM
from src.tools.mysql_client import MySQLClient
from src.tools.qdrant_client import QdrantVectorTool
from src.tools.mongo_client import MongoDBClient

class SQLNode():
    def __init__(self):
        self.llm = BedrockLLM()
        self.mysql_client = MySQLClient(
            os.environ['MYSQL_SERVER'], 
            os.environ['MYSQL_PORT'], 
            os.environ['MYSQL_PASSWORD']
        )
        self.qdrant_client = QdrantVectorTool(url="localhost", port=6333, collection_name="client_ddl")
        self.mongo_client = MongoDBClient(
            uri=os.environ['MONGO_URI'], 
            db_name=os.environ['MONGO_DB']
        )
        # self.schema = self.mysql_client.get_table_info()
    
    def get_schema(self, question: str) -> str:
        docs = self.qdrant_client.search_embedding(question)
        description = ""
        for item in docs:
            description += item.payload["description"] + "\n"
        
        system_message = SystemMessage(content=f"""You are a database expert.
        Given a user question and a list of table descriptions, select only the table(s) that are most relevant to answering the question.
        Respond with a **Python list of table names**, e.g. ["orders", "products"].  
        Do not include any explanation or extra text.
        If no table is relevant, return an empty list [].""")
        human_message = HumanMessage(content=f"User question: {question}\nTable descriptions: {description}")
        
        messages = [system_message, human_message]
        data = self.llm.invoke(messages)
        table_names = json.loads(data.content)
        schema = ""
        for name in table_names:
            dll = self.qdrant_client.get_dll_by_name(name)
            schema += dll + "\n"
        return schema
    
    def generate_sql_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a SQL query based on the user's question.
        Args:
            state (GraphState): The current graph state containing the question
        Returns:
            Dict[str, Any]: Dictionary containing the generated SQL query and updated history
        """
        # schema = self.get_schema(state["question"])
        
        history = state.get("history", [])
        question = state["question"]
        history.extend([f"Human: {question}"])
        
        docs = self.qdrant_client.search_hybrid(question, limit=10)
        schema = ""
        for item in docs:
            schema += item.payload["dll"] + "\n"
        # print("==> schema: \n", schema)
        # Create system and human messages
        system_message = SystemMessage(content=f"Based on the table schema below, write a SQL query that would answer the chat history. Only return the SQL query, no other text like ```sql.\n\nSchema:\n{schema}")
        human_message = HumanMessage(content="\n".join(history))
        
        # Create messages list for the conversation
        messages = [system_message, human_message]
        
        # Invoke LLM with messages
        sql_response = self.llm.invoke(messages)
        
        print("==> SQL query tokens: ", sql_response.usage_metadata)
        print("==> SQL query: ", sql_response.content)
        # Update history with the conversation
        history.extend([
            f"SQL Query: {sql_response.content}"
        ])
        
        return {
            "sql_query": sql_response.content,
            "history": history
        }
    
    def execute_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute the SQL query and return the result.
        """
        try:
            result = self.mysql_client.run_query(state["sql_query"])
            history = state.get("history", [])
            history.extend([
                f"SQL Query Result: {result}"
            ])
            return {"sql_retriever": result, "history": history}
        
        except Exception as e:
            return {"sql_retriever": f"Error executing query: {str(e)}", "loop_step": state.get("loop_step", 0) + 1}
    
    
    def handle_sql_error(self, state: GraphState) -> str:
        """
        Callback function to handle SQL query errors.
        Uses LLM to edit the query and retry up to 3 times.
        """ 
        # Check if we've exceeded max retries
        max_retries = state.get("max_retries", 3)
        if state.get("loop_step", 0) >= max_retries:
            state['loop_step'] = 0
            return "max_retries"
        
        # Check if there was an error in the query result
        query_result = state.get("sql_retriever", "")
        
        if query_result and query_result.startswith("Error executing query:"):
            # Use LLM to fix the SQL query
            error_message = query_result
            original_query = state.get("sql_query", "")
            
            # Create system and human messages
            system_message = SystemMessage(content="Please fix the SQL query to resolve the error. Only return the corrected SQL query, no other text like ```sql.")
            human_message = HumanMessage(content=f"The following SQL query failed with error: {error_message}\n\nOriginal query: {original_query}")
            
            # Create messages list for the conversation
            messages = [system_message, human_message]
            
            try:
                fixed_query = self.llm.invoke(messages)
                # Update the state with the fixed query
                state["sql_query"] = fixed_query.content
                return "error"  # Retry with the fixed query
            except Exception as e:
                return "max_retries"  # If fixing fails, give up
        
        # If no error, proceed to success
        return "success"
    
    def synthetic_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        Callback function to handle SQL query success.
        """
        # Create system and human messages
        chat_history = "\n".join(state['history'])
        system_message = SystemMessage(content="You are insightScanX AI that answers the question based on the chat history")
        human_message = HumanMessage(content=f"Chat History:\n{chat_history}\n Answer the question clearly and concisely based on the chat history.")
        
        # Create messages list for the conversation
        messages = [system_message, human_message]
        # Invoke LLM with messages
        answer = self.llm.invoke(messages)
        
        print("==> answer tokens: ", answer.usage_metadata)
        
        # Update history with the conversation
        history = state.get("history", [])
        history.extend([
            f"Assistant: {answer.content}"
        ])
        
        # Keep only the latest 4 values in the history list
        history = history[-4:] if len(history) > 4 else history
        
        document = {
            "timestamp": datetime.datetime.now(),
            "user_id": "12345",
            "question": state["question"],
            "sql_query": state["sql_query"],
            "answer": answer.content
        }
        
        self.mongo_client.insert_document("chat_collection", document)
        
        return {
            "answer": answer.content,
            "history": history
        }