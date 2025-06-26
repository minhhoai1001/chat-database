import json
from langchain_core.messages import HumanMessage, SystemMessage

from src.state import GraphState
from src.llm import BedrockLLM

# Prompt
router_prompt = """
You are an intelligent routing agent that decides whether a user question should be answered using a SQL database or a knowledge.

- Use **SQL** when the question is related to: **projects**, **scans**, **issues**, or **facilities** — especially if it involves structured data, records, metrics, or anything that requires querying the database.
- Use the **knowledge** when the question is about the **product InsightScanX**, such as how it works, how to use the app, its features, benefits, or user guidance.

Return a JSON object with format above. Do not include any explanation.
{
    "datasource": "sql" or "knowledge"
}
"""

class Router():
    def __init__(self):
        self.llm = BedrockLLM()

    def route(self, state: GraphState) -> str:
        messages = [
            SystemMessage(content=router_prompt),
            HumanMessage(content=state["question"])
        ]
        response = self.llm.invoke(messages)
        try:
            content = json.loads(response.content)
            return content["datasource"]
        except json.JSONDecodeError:
            return "knowledge"
