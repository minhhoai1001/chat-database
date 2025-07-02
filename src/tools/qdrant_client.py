import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.llm import QdrantEmbeddings

class QdrantVectorTool:
    def __init__(self, url: str, port: int, collection_name: str):
        self.client = QdrantClient(host=url, port=port)
        self.collection_name = collection_name
        self.create_collection(collection_name)
        
        self.embeddings = QdrantEmbeddings()

    def create_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def add_embedding(self, name: str, description: str, dll: str):
        embed = self.embed_query(description)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        "name": name,
                        "dll": dll,
                        "description": description,
                    },
                    vector=embed,
                ),
            ],
        )

    def embed_query(self, text: str):
        return self.embeddings.embed_query(text)
    
    def search_embedding(self, query: str):
        data = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embed_query(query),
            limit=5
        )
        return data
    
    def get_dll_by_name(self, name: str):
        filter_by_name = Filter(
            must=[
                FieldCondition(
                    key="name",  # this is the key in your payload
                    match=MatchValue(value=name)
                )
            ]
        )

        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_by_name,
        )
        if len(results) > 0:
            return results[0].payload["dll"]
        else:
            return ""
    
if __name__ == "__main__":
    drant_client = QdrantClient(url="localhost", port=6333, collection_name="test")
    print(drant_client.embed_query("Hello, world!"))