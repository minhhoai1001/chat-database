from datetime import datetime
from pymongo import MongoClient

class MongoDBClient:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="my_database"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_all_collections(self):
        return self.db.list_collection_names()

    def delete_collection(self, collection_name):
        if collection_name in self.db.list_collection_names():
            self.db.drop_collection(collection_name)
            return f"Collection '{collection_name}' deleted."
        else:
            return f"Collection '{collection_name}' does not exist."
    
    def get_collection(self, collection_name):
        return self.db[collection_name]
    
    def insert_document(self, collection_name, document: dict):
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return result.inserted_id
    
    def find_documents(self, collection_name, query: dict):
        collection = self.get_collection(collection_name)
        return list(collection.find(query))
    
    def get_all_user_ids(self, collection_name):
        collection = self.get_collection(collection_name)

        # Get distinct user_id values
        user_ids = collection.distinct("user_id")
    
        return user_ids
    
    def get_user_docs_in_time(
        self,
        collection_name,
        start_time: datetime,
        end_time: datetime,
    ):
        collection = self.get_collection(collection_name)

        query = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }

        results = list(collection.find(query))
        return results

if __name__ == "__main__":
    mongo_client = MongoDBClient()
    
    history = {
        "timestamp": "2023-10-01T12:00:00Z",
        "user_id": "12345",
        "question": "What are the active projects?",
        "history": []
    }
    
    # mongo_client.delete_collection("chat_history")
    # id = mongo_client.insert_document("chat_history", history)
    
    # # print(f"Inserted document with ID: {id}")
    # data = mongo_client.find_documents("chat_collection", {"user_id": "12345"})
    # data = mongo_client.get_all_user_ids("chat_collection")
    # print("Chat History:", data)
    
    user_id = "12345"
    start = datetime(2025, 7, 23, 0, 0, 0)
    end   = datetime(2025, 7, 23, 16, 20, 38)

    docs = mongo_client.get_user_docs_in_time("chat_collection", start, end)
    for doc in docs:
        print(doc)