import streamlit as st
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

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
        chunk_count = collection.count()
        if chunk_count > 0:
            st.success(f"✅ Loaded existing database: {chunk_count} chunks")
            return pipeline
    except:
        pass
    
    # Need to index
    with st.spinner("🔄 First run - indexing documents (30 sec)..."):
        try:
            # Add detailed logging
            from src.document_loader import load_all_documents
            docs = load_all_documents("data/documents")
            st.info(f"📄 Loaded {len(docs)} documents")
            
            if len(docs) == 0:
                st.error("❌ No documents found in data/documents!")
                st.stop()
            
            count = pipeline.index_documents("data/documents")
            st.success(f"✅ Indexed {count} chunks!")
        except Exception as e:
            st.error(f"❌ Indexing failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    return pipeline

st.title("🧠 Enterprise Knowledge Navigator")

pipeline = get_pipeline()

question = st.text_input("💬 Ask:", placeholder="What is SAP's AI policy?")

if st.button("🔍 Search", type="primary"):
    if question:
        with st.spinner("Searching..."):
            result = pipeline.query(question, top_k=3)
            st.markdown("### Answer")
            st.write(result['answer'])
            st.markdown("### Sources")
            for s in result['sources']:

                st.caption(f"{s['source']} ({s['similarity']:.0%})")
