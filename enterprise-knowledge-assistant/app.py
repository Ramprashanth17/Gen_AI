import streamlit as st
from pathlib import Path
import os
import sys

# Get the directory where app.py is located
APP_DIR = Path(__file__).parent
os.chdir(APP_DIR)  # Change to app directory
sys.path.insert(0, str(APP_DIR))

from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Enterprise Knowledge Navigator", page_icon="🧠", layout="wide")

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    
    try:
        collection = pipeline.vector_store.client.get_collection("sap_knowledge")
        if collection.count() > 0:
            return pipeline
    except:
        pass
    
    # Index documents
    st.info("🔄 Indexing documents...")
    pipeline.index_documents("data/documents")
    return pipeline

st.title("🧠 Enterprise Knowledge Navigator")
st.markdown("---")

pipeline = get_pipeline()

question = st.text_input("💬 Ask a question:", placeholder="What is SAP's AI policy?")

if st.button("🔍 Search", type="primary"):
    if question:
        with st.spinner("Searching..."):
            result = pipeline.query(question, top_k=3)
            st.markdown("### Answer")
            st.write(result['answer'])
            st.markdown("### Sources")
            for s in result['sources']:
                st.caption(f"{s['source']} ({s['similarity']:.0%})")