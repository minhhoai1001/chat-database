import os
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import GraphState
from src.llm import BedrockLLM
from src.tools.mysql_client import MySQLClient

sql_prompt = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
Only return the SQL query, no other text like ```sql.
"""

class SQLNode():
    def __init__(self):
        self.llm = BedrockLLM()
        self.mysql_client = MySQLClient(
            os.environ['MYSQL_SERVER'], 
            os.environ['MYSQL_PORT'], 
            os.environ['MYSQL_PASSWORD']
        )
        self.schema = self.mysql_client.get_table_info()
    
    def generate_sql_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Generate a SQL query based on the user's question.
        
        Args:
            question (str): The user's question
            
        Returns:
            str: Generated SQL query
        """
        question = state["question"]
        prompt_template = sql_prompt.format(schema=self.schema, question=question)
        sql_response = self.llm.invoke(prompt_template)
        
        return {"sql_query": sql_response.content}
    
    def execute_query(self, state: GraphState) -> Dict[str, Any]:
        """
        Execute the SQL query and return the result.
        """
        try:
            result = self.mysql_client.run_query(state["sql_query"])
            return {"sql_retriever": result}
        
        except Exception as e:
            return {"sql_retriever": f"Error executing query: {str(e)}", "loop_step": state.get("loop_step", 0) + 1}
    
    
    def handle_sql_error(self, state: GraphState) -> str:
        """
        Callback function to handle SQL query errors.
        Uses LLM to edit the query and retry up to 3 times.
        """ 
        # Check if we've exceeded max retries
        if state.get("loop_step", 0) >= 3:
            state['loop_step'] = 0
            return "max_retries"
        
        # Check if there was an error in the query result
        query_result = state.get("sql_retriever", "")
        print("==> query_result: ", query_result)
        
        if query_result and query_result.startswith("Error executing query:"):
            # Use LLM to fix the SQL query
            error_message = query_result
            original_query = state.get("sql_query", "")
            
            fix_prompt = f"""
            The following SQL query failed with error: {error_message}
            
            Original query: {original_query}
            
            Please fix the SQL query to resolve the error. Only return the corrected SQL query, no other text like ```sql.
            """
            
            try:
                fixed_query = self.llm.invoke(fix_prompt)
                # Update the state with the fixed query
                state["sql_query"] = fixed_query.content
                print("==> fixed_query: ", state["sql_query"])
                return "error"  # Retry with the fixed query
            except Exception as e:
                return "max_retries"  # If fixing fails, give up
        
        # If no error, proceed to success
        return "success"
    
    def synthetic_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        Callback function to handle SQL query success.
        """
        question = state["question"]
        sql_retriever = state["sql_retriever"]
        answer_prompt = f"""
        Based on the SQL query result below, answer the user's question:    
        {sql_retriever}
        
        Question: {question}
        Answer the question shorly and concisely.
        """
        answer = self.llm.invoke(answer_prompt)
        return {"answer": answer.content}