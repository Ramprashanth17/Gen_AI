"""ChromaDB vector database operations"""

import chromadb
from .config import VECTOR_DB_DIR, COLLECTION_NAME

class VectorStore:
    def __init__(self, persist_directory=VECTOR_DB_DIR):
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def get_or_create_collection(self, name=COLLECTION_NAME, metadata=None):
        """Get existing or create new collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(name, metadata=metadata or {})
    
    def add_documents(self, collection_name, documents, embeddings, metadatas):
        """Add documents to collection"""
        collection = self.get_or_create_collection(collection_name)
        
        ids = [f"chunk_{i}" for i in range(len(documents))]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return collection.count()
    
    def query(self, collection_name, query_embedding, n_results=5):
        """Query collection for similar documents"""
        collection = self.client.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return results