import boto3
import numpy as np
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings
from fastembed import TextEmbedding
from langchain.embeddings.base import Embeddings
from FlagEmbedding import BGEM3FlagModel

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

class BedrockEmbeddings(Embeddings):
    def __init__(self, model_id="amazon.titan-embed-text-v1"):
        self.model_id = model_id
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock_client,
            model_id=self.model_id
        )
        
    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, documents: list[str]):
        return self.embeddings.embed_documents(documents)
    
class QdrantEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = TextEmbedding(model_name=model_name)
        
    def embed_query(self, text: str) -> list[float]:
        return list(self.model.embed(text))[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self.model.embed(text))[0] for text in texts]
    
class HybridEmbeddings():
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = BGEM3FlagModel(model_name, use_fp16=True, force_download=True)
        
    def embed_query(self, text: str):
        return self.model.encode(text, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    
    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=True)