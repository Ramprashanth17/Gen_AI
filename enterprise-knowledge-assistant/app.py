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

# At the very top of app.py, add:
import os
from pathlib import Path

st.write(f"🔍 DEBUG: Current directory: {os.getcwd()}")
st.write(f"🔍 DEBUG: Files in current dir: {list(Path('.').glob('*'))[:10]}")
st.write(f"🔍 DEBUG: data/documents exists? {Path('data/documents').exists()}")

if Path('data/documents').exists():
    st.write(f"🔍 DEBUG: Files in data/documents: {list(Path('data/documents').rglob('*'))[:10]}")
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
