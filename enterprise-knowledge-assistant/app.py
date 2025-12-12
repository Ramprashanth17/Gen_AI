import streamlit as st
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Navigator", page_icon="🧠", layout="wide")

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    
    # Check if database exists
    try:
        collection = pipeline.vector_store.client.get_collection("sap_knowledge")
        chunk_count = collection.count()
        if chunk_count == 0:
            raise Exception("Empty collection")
    except:
        # Database doesn't exist - index documents
        st.info("First run detected. Indexing documents... (this takes ~30 seconds)")
        pipeline.index_documents("data/documents")
    
    return pipeline

st.title("🧠 Enterprise Knowledge Navigator")
st.markdown("**SAP Employee Knowledge Assistant** | Powered by RAG")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("System Info")
    try:
        pipeline = get_pipeline()
        collection = pipeline.vector_store.client.get_collection("sap_knowledge")
        st.metric("Knowledge Chunks", collection.count())
    except Exception as e:
        st.error(f"Database error: {str(e)}")

# Main
question = st.text_input("Ask a question:", placeholder="e.g., What is SAP's AI policy?")

if st.button("Search", type="primary") or question:  # Enter key works now!
    if question:
        with st.spinner("Searching..."):
            try:
                pipeline = get_pipeline()
                result = pipeline.query(question, top_k=3)
                
                st.markdown("### 📝 Answer")
                st.success(result['answer'])
                
                st.markdown("### 📚 Sources")
                for source in result['sources']:
                    st.caption(f"📄 {source['source']} (Similarity: {source['similarity']:.1%})")
            except Exception as e:
                st.error(f"Error: {str(e)}")