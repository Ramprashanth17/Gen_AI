import streamlit as st
from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Enterprise Knowledge Navigator",
    page_icon="🧠",
    layout="wide"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# Header
st.title("🧠 Enterprise Knowledge Navigator")
st.markdown("**SAP Employee Knowledge Assistant** | Powered by RAG")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    try:
        collection = pipeline.vector_store.client.get_collection("sap_knowledge")
        chunk_count = collection.count()
        st.metric("Documents Indexed", "8")
        st.metric("Knowledge Chunks", chunk_count)
        st.metric("Embedding Model", "MiniLM-L6-v2")
    except:
        st.warning("Database not initialized")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This system uses Retrieval-Augmented Generation (RAG) to answer questions from SAP policy documents.")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "💬 Ask a question:",
        placeholder="e.g., What is SAP's policy on accepting gifts?",
        help="Ask any question about SAP policies and procedures"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

if search_button and question:
    with st.spinner("🔎 Searching knowledge base..."):
        try:
            result = pipeline.query(question, top_k=3)
            
            # Display answer
            st.markdown("### 📝 Answer")
            st.success(result['answer'])
            
            # Display sources
            st.markdown("### 📚 Sources")
            for i, source in enumerate(result['sources']):
                with st.expander(f"📄 {source['source']} - Chunk {source['chunk_id']+1} (Similarity: {source['similarity']:.1%})"):
                    st.caption(f"Relevance: {source['similarity']:.1%}")
            
            # Stats
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks Retrieved", result['chunks_used'])
            col2.metric("Top Similarity", f"{result['sources'][0]['similarity']:.1%}")
            col3.metric("Response Length", f"{len(result['answer'])} chars")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try rephrasing your question")

elif search_button:
    st.warning("⚠️ Please enter a question")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • OpenAI GPT-4o-mini • ChromaDB • Sentence Transformers")