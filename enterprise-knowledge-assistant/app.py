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

# Initialize session state
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = DEFAULT_COLLECTION

@st.cache_resource
def get_pipeline():
    pipeline = RAGPipeline()
    
    for kb_name, kb_config in COLLECTIONS.items():
        collection_name = kb_config["name"]
        folder_path = kb_config["folder"]
        
        try:
            col = pipeline.vector_store.client.get_collection(collection_name)
            if col.count() == 0:
                raise Exception("Empty")
        except:
            st.info(f"📦 Indexing {kb_name} documents...")
            pipeline.index_documents(folder_path, collection_name)
    
    return pipeline

pipeline = get_pipeline()

st.title("🧠 Enterprise Knowledge Navigator")
st.markdown("Multi-tenant RAG for Enterprise Documentation")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    knowledge_base = st.selectbox(
        "📚 Knowledge Base:",
        list(COLLECTIONS.keys()),
        index=list(COLLECTIONS.keys()).index(st.session_state.knowledge_base),
        key="kb_select"
    )
    
    st.session_state.knowledge_base = knowledge_base
    selected_collection = COLLECTIONS[knowledge_base]["name"]
    
    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    
    examples = {
        "SAP": [
            "What is SAP's policy on accepting gifts?",
            "What are the AI ethics principles at SAP?",
            "Who oversees AI ethics at SAP?"
        ],
        "Salesforce": [
            "What is Salesforce Apex?",
            "What are SOQL governor limits?",
            "How do you handle DML in loops?"
        ]
    }
    
    for i, example in enumerate(examples[knowledge_base]):
        if st.button(example, key=f"ex_{knowledge_base}_{i}", use_container_width=True):
            st.session_state.question = example
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 System Info")
    
    try:
        # Use selected_collection (not a cached variable!)
        current_collection = pipeline.vector_store.client.get_collection(selected_collection)
        count = current_collection.count()
        
        st.metric("Chunks Indexed", count)
        st.caption(f"Collection: {selected_collection}")  # Show which collection!
        
        #col = pipeline.vector_store.client.get_collection(selected_collection)
        #st.metric("Chunks Indexed", count)
        
        st.markdown("### 📄 Documents")
        docs = set()
        results = current_collection.get(limit=2000)
        if results and results['metadatas']:
            for meta in results['metadatas']:
                docs.add(meta['source'])
        
            for doc in sorted(docs):
                st.caption(f"📄 {doc}")
        else:
            st.caption("No documents found")
    except Exception as e:
        st.error(f"Error loading collection: {str(e)}")

# Main
st.markdown(f"### 💬 Ask about {knowledge_base}")

question = st.text_input(
    "Your question:",
    value=st.session_state.question,
    placeholder=f"e.g., What is {knowledge_base}'s policy?",
    key="q_input"
)

if st.button("🔍 Search", type="primary") or (question and question != st.session_state.get('last_question', '')):
    if question:
        st.session_state.last_question = question
        st.session_state.question = ""  # Clear for next query
        
        with st.spinner(f"Searching {knowledge_base}..."):
            result = pipeline.query(question, collection_name=selected_collection, top_k=3)
            
            st.markdown("### 📝 Answer")
            st.success(result['answer'])
            
            st.markdown("### 📚 Sources")
            for s in result['sources']:
                st.caption(f"📄 {s['source']} | Chunk {s['chunk_id']+1} | Similarity: {s['similarity']:.1%}")
    else:
        st.warning("Please enter a question")

st.markdown("---")
st.caption(f"Searching: **{knowledge_base}** knowledge base")