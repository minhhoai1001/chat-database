import gradio as gr
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver

from src.llm import BedrockLLM
from src.nodes.sql import SQLNode
from src.nodes.knowledge import KnowledgeNode
from src.conditional.router import Router
from src.state import GraphState

load_dotenv()

class ChatDatabaseAgent:
    def __init__(self, model_id="us.meta.llama4-maverick-17b-instruct-v1:0", temperature=0.1, max_tokens=512):
        self.llm = BedrockLLM(model_id=model_id, temperature=temperature, max_tokens=max_tokens)
        self.sql_node = SQLNode()
        self.knowledge_node = KnowledgeNode()
        self.router = Router()
        self.workflow = StateGraph(GraphState)
        self.memory = MemorySaver()
        self.build_graph()
    
    
    def build_graph(self):
        self.workflow.add_node("sqlGenerateAgent", self.sql_node.generate_sql_query)
        self.workflow.add_node("sqlExecuteAgent", self.sql_node.execute_query)
        self.workflow.add_node("knowledgeAgent", self.knowledge_node.run)
        self.workflow.add_node("syntheticAnswerAgent", self.sql_node.synthetic_answer)
        
        self.workflow.set_conditional_entry_point(
            self.router.route,
            {
                "sql": "sqlGenerateAgent",
                "knowledge": "knowledgeAgent",
            }
        )
        
        self.workflow.add_edge("sqlGenerateAgent", "sqlExecuteAgent")
        self.workflow.add_edge("sqlExecuteAgent", "syntheticAnswerAgent")
        
        self.workflow.add_conditional_edges(
            "sqlExecuteAgent",
            self.sql_node.handle_sql_error,
            {
                "error": "sqlExecuteAgent",
                "max_retries": END,
                "success": "syntheticAnswerAgent",
            }
        )
        
        self.graph = self.workflow.compile(checkpointer=self.memory)
        
    def run(self, state: GraphState, thread_id:str = "1"):
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(state, config)
    

    def respond(self, message, history):
        """
        Main response method for the chat interface.
        
        Args:
            message (str): User's message
            history (list): Conversation history
            
        Yields:
            str: Streaming response
        """
        try:
            response = self.run({"question": message})
            text = ""
            for key, item in response.items():
                text += f"{key}: {item}\n"
                
                yield text
            
        except Exception as e:
            yield f"Error: {str(e)}"


# Initialize the agent
agent = ChatDatabaseAgent()
# Save the graph visualization to a PNG file
with open("graph.png", "wb") as f:
    f.write(agent.graph.get_graph().draw_mermaid_png())


# Create the Gradio interface
demo = gr.ChatInterface(
    fn=agent.respond,
    title="Amazon Bedrock Chat Database Agent",
    description="Chat interface using Amazon Bedrock with LLAMA 4 for database queries",
    examples=[
        ["How many projects are active?"],
        ["What are the recent exports?"],
        ["Count the total number of records"],
    ],
    theme="soft"
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()