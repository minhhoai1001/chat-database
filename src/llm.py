import boto3
from langchain_aws import ChatBedrockConverse

class LLM:
    def invoke(self, prompt):
        raise NotImplementedError

    def stream(self, prompt):
        raise NotImplementedError

class BedrockLLM(LLM):
    def __init__(self, model_id="us.meta.llama4-maverick-17b-instruct-v1:0", temperature=0.1, max_tokens=512):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1' 
        )
        
        # Initialize BedrockChat
        self.llm = ChatBedrockConverse(
            client=self.bedrock_client,
            model_id=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
    def invoke(self, prompt):
        return self.llm.invoke(prompt)
    
    def stream(self, prompt):
        return self.llm.stream(prompt)
    