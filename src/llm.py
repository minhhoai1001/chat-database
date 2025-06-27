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
        # Handle both string prompts and message lists
        if isinstance(prompt, str):
            return self.llm.invoke(prompt)
        elif isinstance(prompt, list):
            # For message lists, we need to convert them to the format expected by Bedrock
            # Convert LangChain messages to Bedrock format
            messages = []
            for msg in prompt:
                if hasattr(msg, 'type') and hasattr(msg, 'content'):
                    if msg.type == 'system':
                        messages.append({"role": "system", "content": msg.content})
                    elif msg.type == 'human':
                        messages.append({"role": "user", "content": msg.content})
                    elif msg.type == 'ai':
                        messages.append({"role": "assistant", "content": msg.content})
                else:
                    # Fallback for other message types
                    messages.append({"role": "user", "content": str(msg)})
            
            return self.llm.invoke(messages)
        else:
            # Fallback for other types
            return self.llm.invoke(str(prompt))
    
    def stream(self, prompt):
        return self.llm.stream(prompt)
    