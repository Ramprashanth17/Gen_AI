import streamlit as st
from pathlib import Path
import os
import sys

APP_DIR = Path(__file__).parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
from src.config import COLLECTIONS, DEFAULT_COLLECTION  

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Navigator", page_icon="🧠", layout="wide")

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    
    # Index SAP if not exists
    try:
        sap_col = pipeline.vector_store.client.get_collection("sap_knowledge")
        if sap_col.count() == 0:
            raise Exception("Empty")
    except:
        st.info("📦 Indexing SAP documents...")
        pipeline.index_documents("data/documents/sap", "sap_knowledge")
    
    # Index Salesforce if not exists
    try:
        sf_col = pipeline.vector_store.client.get_collection("salesforce_knowledge")
        if sf_col.count() == 0:
            raise Exception("Empty")
    except:
        st.info("📦 Indexing Salesforce documents...")
        pipeline.index_documents("data/documents/salesforce", "salesforce_knowledge")
    
    return pipeline

# Header
st.title("🧠 Enterprise Knowledge Navigator")
st.markdown("Multi-tenant RAG for Enterprise Documentation")
st.markdown("---")

# Sidebar - Knowledge Base Selector
with st.sidebar:
    st.header("⚙️ Settings")
    
    knowledge_base = st.selectbox(
    "📚 Knowledge Base:",
    list(COLLECTIONS.keys()),  # Dynamic from config!
    index=list(COLLECTIONS.keys()).index(DEFAULT_COLLECTION)
)

    selected_collection = COLLECTIONS[knowledge_base]["name"]
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    pipeline = get_pipeline()
    try:
        col = pipeline.vector_store.client.get_collection(selected_collection)
        st.metric("Chunks Indexed", col.count())

        # Show documents in this collection
        st.markdown("### 📄 Documents")
        docs_in_collection = set()

        # Get unique sources from collection
        results = col.get(limit=1000)  # Get metadata from collection
        for metadata in results['metadatas']:
            docs_in_collection.add(metadata['source'])
        
        # Display as list
        for doc in sorted(docs_in_collection):
            st.caption(f"📄 {doc}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Main interface
st.markdown(f"### 💬 Ask a question about {knowledge_base} policies")

question = st.text_input(
    "Your question:",
    placeholder=f"e.g., What is {knowledge_base}'s policy on gifts?" if knowledge_base == "SAP" else f"e.g., What are {knowledge_base} user permissions?"
)

if st.button("🔍 Search", type="primary"):
    if question:
        with st.spinner(f"Searching {knowledge_base} knowledge base..."):
            result = pipeline.query(question, collection_name=selected_collection, top_k=3)
            
            st.markdown("### 📝 Answer")
            st.success(result['answer'])
            
            st.markdown("### 📚 Sources")
            for s in result['sources']:
                st.caption(f"📄 {s['source']} | Chunk {s['chunk_id']+1} | Similarity: {s['similarity']:.1%}")

st.markdown("---")
st.caption(f"Currently searching: **{knowledge_base}** knowledge base")