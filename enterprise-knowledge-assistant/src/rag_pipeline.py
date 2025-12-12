"""Complete RAG pipeline"""

import os
from openai import OpenAI
from .config import *
from .document_loader import load_all_documents
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingManager()
        self.vector_store = VectorStore()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection_name = COLLECTION_NAME
    
    def index_documents(self, folder_path):
        """Load, chunk, embed, and store all documents"""
        print(f"Loading documents from {folder_path}...")
        documents = load_all_documents(folder_path)
        
        all_chunks = []
        chunk_metadata = []
        
        for doc in documents:
            chunks = self.embedder.chunk_text(doc['text'])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'source': doc['metadata']['source'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'doc_type': doc['metadata']['type']
                })
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedder.generate_embeddings(all_chunks)
        
        print(f"Storing in vector database...")
        count = self.vector_store.add_documents(
            self.collection_name,
            all_chunks,
            embeddings.tolist(),
            chunk_metadata
        )
        
        print(f"✅ Indexed {count} chunks!")
        return count
    
    def query(self, question, top_k=TOP_K_RESULTS):
        """Query the knowledge base and generate answer"""
        # Embed query
        query_embedding = self.embedder.model.encode([question])[0]
        
        # Retrieve relevant chunks
        results = self.vector_store.query(
            self.collection_name,
            query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format context
        context_parts = []
        sources = []
        
        for i in range(len(results['documents'][0])):
            chunk_text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            context_parts.append(
                f"[Source: {metadata['source']}, Chunk {metadata['chunk_id']+1}]\n{chunk_text}"
            )
            sources.append({
                'source': metadata['source'],
                'chunk_id': metadata['chunk_id'],
                'similarity': 1 - distance
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        response = self.openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        return {
            'answer': response.choices[0].message.content,
            'sources': sources,
            'chunks_used': len(results['documents'][0])
        }