import uuid
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.llm import QdrantEmbeddings, HybridEmbeddings

class QdrantVectorTool:
    def __init__(self, url: str, port: int, collection_name: str):
        self.client = QdrantClient(host=url, port=port)
        self.collection_name = collection_name
        self.create_collection(collection_name)
        self.hybrid_embeddings = HybridEmbeddings("BAAI/bge-m3")

    def create_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE,
                    ),
                    "colbert": models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM), # use token-level max similarity
                        hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for better reranking control
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                }
            )
        
    def add_embedding(self, name: str, description: str, dll: str):
        embeddings = self.hybrid_embeddings.embed_query(description)
        dense_vecs = embeddings['dense_vecs']
        sparse_vecs = embeddings['lexical_weights']
        colbert_vecs = embeddings['colbert_vecs']
        sparse_vecs = {
            "indices": [int(key) for key in sparse_vecs.keys()],
            "values": [float(value) for value in sparse_vecs.values()],
        }
        vector = {
            "dense": dense_vecs.tolist(),
            "sparse": sparse_vecs,
            "colbert": colbert_vecs.tolist(),
        }
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
                    vector=vector,
                ),
            ],
        )
    
    def search_embedding(self, query: str, limit: int = 5):
        embeddings = self.hybrid_embeddings.embed_query(query)
        dense_vecs = embeddings['dense_vecs']
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vecs.tolist(),
            using="dense",
            with_payload=True,
            limit=limit
        )
        return results.points
    
    def search_hybrid(self, query: str, limit: int = 5):
        embeddings = self.hybrid_embeddings.embed_query(query)
        dense_vecs = embeddings['dense_vecs']
        sparse_vecs = embeddings['lexical_weights']
        
        sparse = {
            "indices": [int(key) for key in sparse_vecs.keys()],
            "values": [float(value) for value in sparse_vecs.values()],
        }
        prefetch = [
            models.Prefetch(
                query=dense_vecs.tolist(),
                using="dense",
                limit=limit,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                using="sparse",
                limit=limit,
            ),
        ]
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True,
            limit=limit,
        )

        return results.points
    
    def search_rerank(self, query: str, limit: int = 5):
        embeddings = self.hybrid_embeddings.embed_query(query)
        dense_vecs = embeddings['dense_vecs']
        sparse_vecs = embeddings['lexical_weights']
        colbert_vecs = embeddings['colbert_vecs']
        
        sparse = {
            "indices": [int(key) for key in sparse_vecs.keys()],
            "values": [float(value) for value in sparse_vecs.values()],
        }
        prefetch = [
            models.Prefetch(
                query=dense_vecs.tolist(),
                using="dense",
                limit=10,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse),
                using="sparse",
                limit=10,
            ),
        ]
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=colbert_vecs.tolist(),
            using="colbert",
            with_payload=True,
            limit=limit,
        )

        return results.points
    
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
    query = "How many issues and scans are there in the most recent project?"
    results = drant_client.search_rerank(query)
    for result in results:
        print(result.payload["name"], result.score)
        print("-"*100)