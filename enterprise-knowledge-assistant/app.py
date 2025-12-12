import streamlit as st
from pathlib import Path
import os
import sys

# Detect where we are and set paths correctly
current_file = Path(__file__).resolve()
app_dir = current_file.parent

# Add to Python path
sys.path.insert(0, str(app_dir))

from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Navigator", page_icon="🧠")

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    
    # Check if collection exists
    try:
        collection = pipeline.vector_store.client.get_collection("sap_knowledge")
        if collection.count() > 0:
            return pipeline
    except:
        pass
    
    # Index documents with absolute path
    docs_path = app_dir / "data" / "documents"
    st.info(f"Indexing from: {docs_path}")
    
    if not docs_path.exists():
        st.error(f"Documents not found at {docs_path}")
        st.stop()
    
    pipeline.index_documents(str(docs_path))
    return pipeline

st.title("🧠 Enterprise Knowledge Navigator")

pipeline = get_pipeline()

question = st.text_input("Ask a question:", placeholder="What is SAP's AI policy?")

if st.button("Search", type="primary"):
    if question:
        with st.spinner("Searching..."):
            result = pipeline.query(question)
            
            st.markdown("### 📝 Answer")
            st.success(result['answer'])
            
            st.markdown("### 📚 Sources")
            for s in result['sources']:
                st.caption(f"📄 {s['source']} (Similarity: {s['similarity']:.0%})")