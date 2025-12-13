import streamlit as st
from pathlib import Path
import os
import sys

APP_DIR = Path(__file__).parent
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))

from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

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
        ["SAP", "Salesforce"],
        help="Select which company's documentation to search"
    )
    
    # Map to collection names
    collection_map = {
        "SAP": "sap_knowledge",
        "Salesforce": "salesforce_knowledge"
    }
    selected_collection = collection_map[knowledge_base]
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    pipeline = get_pipeline()
    try:
        col = pipeline.vector_store.client.get_collection(selected_collection)
        st.metric("Chunks Indexed", col.count())
    except:
        st.metric("Chunks Indexed", "N/A")

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