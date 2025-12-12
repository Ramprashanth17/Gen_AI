"""Embedding generation using sentence transformers"""

from sentence_transformers import SentenceTransformer
import tiktoken
from .config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

class EmbeddingManager:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        """Split text into overlapping chunks"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_str = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_str)
            start = start + chunk_size - overlap
            if end >= len(tokens):
                break
        
        return chunks
    
    def generate_embeddings(self, texts, batch_size=32):
        """Generate embeddings for list of texts"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)