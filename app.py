import gradio as gr
import random
import time
import boto3
from langchain_aws import ChatBedrockConverse
from langchain.schema import HumanMessage, AIMessage

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1' 
)

# Initialize BedrockChat
llm = ChatBedrockConverse(
    client=bedrock_client,
    model_id="us.meta.llama4-maverick-17b-instruct-v1:0",
    temperature=0.1,
    max_tokens=512
)

def format_history(history):
    formatted_history = []
    for human, ai in history:
        formatted_history.append(HumanMessage(content=human))
        if ai:
            formatted_history.append(AIMessage(content=ai))
    return formatted_history

def respond(message, history):
    try:
        messages = format_history(history)
        messages.append(HumanMessage(content=message))

        partial_response = ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                chunk_content = chunk.content
                if isinstance(chunk_content, list):
                    # Safely extract string parts from list of dicts or strings
                    chunk_content = ''.join(
                        item["text"] if isinstance(item, dict) and "text" in item else str(item)
                        for item in chunk_content
                    )
            else:
                chunk_content = str(chunk)

            partial_response += chunk_content
            yield partial_response

    except Exception as e:
        yield f"Error: {str(e)}"

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=respond,
    title="Amazon Bedrock Chat",
    description="Chat interface using Amazon Bedrock with LLAMA 4",
    examples=[
        ["What is quantum computing?"],
        ["Explain machine learning in simple terms"],
        ["What are the best practices for cloud security?"],
    ],
    theme="soft"
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()