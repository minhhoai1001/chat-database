from typing import Any, Dict
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import GraphState
from src.llm import BedrockLLM

knowledge_prompt = """
You are an expert at answering questions about the product InsightScanX.
"""

class KnowledgeNode():
    def __init__(self):
        self.llm = BedrockLLM()
        
    def run(self, state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        messages = [
            SystemMessage(content=knowledge_prompt),
            HumanMessage(content=question)
        ]
        response = self.llm.invoke(messages)
        return {"answer": response.content}