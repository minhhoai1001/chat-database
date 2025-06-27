import os
from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import GraphState
from src.llm import BedrockLLM
from src.tools.mysql_client import MySQLClient

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
            state (GraphState): The current graph state containing the question
            
        Returns:
            Dict[str, Any]: Dictionary containing the generated SQL query and updated history
        """
        history = state.get("history", [])
        question = state["question"]
        history.extend([
            f"Human: {question}"
        ])
        # Create system and human messages
        system_message = SystemMessage(content=f"Based on the table schema below, write a SQL query that would answer the chat history. Only return the SQL query, no other text like ```sql.\n\nSchema:\n{self.schema}")
        human_message = HumanMessage(content="\n".join(history))
        
        # Create messages list for the conversation
        messages = [system_message, human_message]
        
        # Invoke LLM with messages
        sql_response = self.llm.invoke(messages)
        
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
        
        # Update history with the conversation
        history = state.get("history", [])
        history.extend([
            f"Assistant: {answer.content}"
        ])
        
        # Keep only the latest 4 values in the history list
        history = history[-4:] if len(history) > 4 else history
        
        return {
            "answer": answer.content,
            "history": history
        }