# 🎓 RAG Learning Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

**An interactive learning platform for mastering Retrieval-Augmented Generation (RAG) systems from first principles to production deployment.**

> *Learn how to build AI systems that ground Large Language Models in factual information, eliminating hallucinations and enabling verifiable, source-cited answers.*

---

## 🌟 What is This?

This project is a **comprehensive educational platform** that teaches you how to build production-ready RAG systems through hands-on tutorials and working implementations. You'll go from understanding basic embeddings to deploying complete question-answering systems grounded in your own documents.

### ✨ Key Features

- 📚 **4 Progressive Tutorials** - Master embeddings, chunking, vector storage, and RAG pipelines
- 🎯 **Hands-On Learning** - Every concept includes runnable code and experiments
- 🧪 **Practice Exercises** - Reinforce learning with real coding challenges
- 🎥 **Video Walkthrough** - 10-minute show-and-tell demonstration
- 💼 **Production-Ready** - Industry-standard techniques (recursive splitting, metadata handling)
- 🚀 **Complete Demo** - Working RAG system with research papers in minutes

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or Jupyter Notebook
- Gemini API key (free tier available)

### Installation
```bash
# Clone the repository
git clone https://github.com/Ramprashanth17/Gen_AI.git
cd Gen_AI/rag-learning-platform

# Install dependencies
pip install -r requirements.txt

# Set up API key (get the free key from https://ai.google.dev/)
# Option 1: Environment variable
export GEMINI_API_KEY='your-api-key-here'

# Option 2: Colab Secrets (recommended for notebooks)
# Add GEMINI_API_KEY in Colab's Secrets manager (🔑 icon)
```

### Run the Demo

**Fastest Way to See RAG in Action:**
```bash
# Open the DEMO notebook
jupyter notebook notebooks/DEMO_full_rag_system.ipynb

# Or use Google Colab:
# 1. Go to https://colab.research.google.com/
# 2. File → Open notebook → GitHub
# 3. Enter: Ramprashanth17/Gen_AI
# 4. Select: rag-learning-platform/notebooks/DEMO_full_rag_system.ipynb
# 5. Run all cells!

# Runtime: ~10 minutes
# Result: Working RAG system answering questions about research papers!
```

---

## 📚 Learning Path

Start here and progress through the tutorials sequentially:

### 1️⃣ [Embeddings Fundamentals](notebooks/1_Embeddings_Fundamentals.ipynb)
**Time:** 40-50 minutes | **Level:** Beginner

Learn how text becomes numbers and why it matters for semantic search.

**What you'll learn:**
- What are embeddings, and how do they capture meaning?
- Vector similarity and cosine distance
- Visualizing embeddings in 2D/3D space
- Why "dog" and "puppy" have similar vectors

**Key concepts:** Semantic similarity, vector representations, cosine similarity

---

### 2️⃣ [Chunking & Tokenization](notebooks/2_Chunking_and_Tokenization.ipynb)
**Time:** 40-50 minutes | **Level:** Intermediate

Master document preprocessing strategies for RAG systems.

**What you'll learn:**
- The difference between chunking, tokenization, and embedding
- Recursive character splitting (industry standard!)
- Optimal chunk sizes and overlap strategies
- Metadata extraction for production systems

**Key concepts:** Document preprocessing, semantic boundaries, token optimization

---

### 3️⃣ [Vector Storage with ChromaDB](notebooks/3_Vector_Storage_Chromadb.ipynb)
**Time:** 40-50 minutes | **Level:** Intermediate

Build searchable vector databases for semantic retrieval.

**What you'll learn:**
- What are vector databases, and why does RAG need them?
- ChromaDB setup and operations
- Storing embeddings with metadata
- Advanced querying with filters

**Key concepts:** Vector databases, similarity search, metadata filtering

---

### 4️⃣ [Complete RAG Pipeline](notebooks/4_RAG_Pipeline_Complete.ipynb)
**Time:** 50-60 minutes | **Level:** Advanced

Connect all components into a production-ready RAG system.

**What you'll learn:**
- Building end-to-end RAG pipelines
- Prompt engineering for grounded answers
- RAG vs baseline LLM comparison
- Error handling and evaluation metrics

**Key concepts:** Retrieval-augmented generation, prompt engineering, system evaluation

---

### 🎯 [DEMO: Full System](notebooks/Demo_RAG_Pipeline_ResearchPapers.ipynb)
**Time:** 10 minutes | **Level:** All levels

See everything working together in a streamlined demo.

**Perfect for:**
- Quick overview before diving into tutorials
- Demonstrating RAG to others
- Understanding the end goal
- Video recording and presentations

---

## 🛠️ Technologies Used

| Technology | Purpose | Why This Choice |
|------------|---------|-----------------|
| **Sentence Transformers** | Generate embeddings | SOTA quality, local execution, free |
| **ChromaDB** | Vector storage & search | Zero config, perfect for learning |
| **Google Gemini** | LLM generation | Generous free tier, good quality |
| **Python** | Implementation | Industry standard for AI/ML |
| **Jupyter** | Interactive learning | Excellent for tutorials |

**Full Stack:**
- `sentence-transformers` - Embeddings (all-MiniLM-L6-v2)
- `chromadb` - Vector database with HNSW indexing
- `google-generativeai` - Gemini 2.5 Flash LLM
- `tiktoken` - Token counting and cost estimation
- `PyPDF2` - PDF text extraction
- `matplotlib` + `pandas` - Visualization and analysis

---

## 📖 What You'll Build

By completing this tutorial series, you'll build:

### 🎯 Academic Research Assistant (DEMO Project)

A complete RAG system that:
- ✅ Processes research papers from PDFs
- ✅ Chunks papers into semantic sections
- ✅ Generates embeddings for semantic search
- ✅ Retrieves relevant passages using vector similarity
- ✅ Generates answers with proper academic citations
- ✅ Compares RAG vs baseline to show the advantage

### 💡 Skills You'll Master

**Technical Skills:**
- Implement semantic search using embeddings
- Design chunking strategies for different document types
- Set up and query vector databases
- Engineer prompts that prevent LLM hallucination
- Build complete RAG pipelines from scratch
- Evaluate and optimize RAG system performance

**Production Skills:**
- Handle API rate limits and costs
- Implement error handling and fallback strategies
- Extract metadata for source attribution
- Design scalable document processing pipelines
- Optimize for latency and cost trade-offs

---

## 🎨 Example Use Cases

**This RAG architecture works for:**

- 📚 **Academic Research** - Literature review assistants, paper Q&A
- 💼 **Enterprise Knowledge** - Company documentation search, policy lookup
- 🏥 **Healthcare** - Medical guideline reference, diagnosis assistance
- ⚖️ **Legal** - Contract analysis, case law search
- 🛠️ **Developer Tools** - API documentation search, code example retrieval
- 🎓 **Education** - Study assistants, textbook Q&A
- 🤝 **Customer Support** - Knowledge base chatbots, ticket automation

**Just change the documents - the RAG pipeline stays the same!**

---

## 📊 Project Structure
```
rag-learning-platform/
│
├── notebooks/              # Interactive tutorials
│   ├── 01_embeddings_fundamentals.ipynb
│   ├── 02_chunking_and_tokenization.ipynb
│   ├── 03_vector_storage_chromadb.ipynb
│   ├── 04_rag_pipeline_complete.ipynb
│   └── DEMO_full_rag_system.ipynb
│
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
└── README.md            # This file
```

---

## 🎓 Learning Outcomes

**After completing this tutorial series, you will be able to:**

✅ Explain how RAG systems prevent LLM hallucination  
✅ Implement semantic search using embeddings and vector databases  
✅ Choose appropriate chunking strategies for your use case  
✅ Build complete RAG pipelines from scratch  
✅ Deploy RAG systems for real-world applications  
✅ Debug and optimize RAG performance  
✅ Understand trade-offs between RAG, fine-tuning, and prompt engineering  

---

## 🌟 Why This Project Stands Out

### 🏆 Production-Ready Techniques

Unlike simplified tutorials, this teaches **industry-standard approaches:**

- ✅ **Recursive Character Splitting** - Same algorithm used by LangChain
- ✅ **Metadata Handling** - Source attribution for verifiable answers
- ✅ **Error Handling** - Graceful degradation and rate limit management
- ✅ **Evaluation Metrics** - Quantitative quality assessment
- ✅ **Cost Optimization** - Token counting and budget management

### 📖 Comprehensive Teaching Materials

Each tutorial includes:
- ✅ Clear concept explanations with analogies
- ✅ Visual diagrams and interactive plots
- ✅ Hands-on code experiments
- ✅ Practice exercises with solutions
- ✅ "Common Doubts & Pitfalls" sections
- ✅ Real-world examples and use cases

### 🎯 Learn by Doing

- 40+ code experiments to run and modify
- 15+ practice exercises across tutorials
- Real PDFs processing (not just toy examples)
- Production debugging scenarios
- Cost and performance optimization challenges

---

## 💻 Usage Example
```python
# Quick example using the DEMO notebook

from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load your documents
documents = ["Your text here...", "More documents..."]

# 2. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# 3. Store in vector database
client = chromadb.Client()
collection = client.create_collection("my_docs")
collection.add(documents=documents, embeddings=embeddings, ids=[...])

# 4. Query
results = collection.query(query_texts=["Your question?"], n_results=3)

# 5. Generate an answer with LLM + context
# (See Tutorial 04 for complete implementation)
```

**Full pipeline in < 20 lines of code!**


## 🤝 Contributing

Found an issue or want to improve the tutorials?

- 🐛 **Report bugs** via GitHub Issues
- 💡 **Suggest improvements** via Pull Requests
- 📧 **Ask questions** via email (below)

**Contributions welcome!** This is an educational project designed to help others learn RAG.

---

## 📈 Project Stats

- **Tutorials:** 4 comprehensive notebooks
- **Code Cells:** 150+ executable examples
- **Exercises:** 15+ practice problems
- **Visualizations:** 20+ plots and diagrams
- **Lines of Code:** ~2,000 (well-commented)
- **Documentation:** Extensive markdown explanations
- **Learning Time:** ~3-4 hours total
- **Demo Runtime:** 10 minutes

---

## 🎯 Who Is This For?

**Perfect for:**
- 🎓 Data science students learning modern AI techniques
- 💼 AI engineers building production RAG systems
- 🔬 Researchers needing literature review tools
- 👨‍💻 Developers integrating LLMs into applications
- 📚 Anyone curious about how ChatGPT plugins and AI assistants work

**Prerequisites:**
- Basic Python programming
- Familiarity with Jupyter notebooks
- Understanding of basic ML concepts (helpful but not required)

---

## 🗺️ Roadmap

**Current Features:**
- ✅ Complete 4-tutorial learning path
- ✅ Working DEMO with research papers
- ✅ RAG vs baseline comparison
- ✅ Comprehensive documentation

**Coming Soon:**
- 🔨 Streamlit web interface for interactive demos
- 🔨 Tutorial 05: Advanced RAG (re-ranking, hybrid search)
- 🔨 Multi-modal RAG (handling images and tables)
- 🔨 Deployment guide (Docker, cloud hosting)

---

## 📚 Additional Resources

**Learn More About RAG:**
- [Original RAG Paper](https://arxiv.org/abs/2005.11401) - Lewis et al., 2020
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)

**Technologies Documentation:**
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://docs.trychroma.com/)
- [Google Gemini](https://ai.google.dev/)

**Related Projects:**
- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLMs
- [Haystack](https://github.com/deepset-ai/haystack) - NLP framework with RAG

---

## 🏆 Project Highlights

### 🎓 Educational Excellence

- **Progressive Complexity:** Start simple (single sentences) → End advanced (multi-paper research)
- **Multi-Modal Learning:** Text explanations + Visual diagrams + Code experiments + Video walkthrough
- **Common Pitfalls Addressed:** Explicit sections covering the confusions learners typically face
- **Real-World Focus:** Production techniques, not just academic exercises

### 💼 Career Relevance

**Skills demonstrated in this project:**
- ✅ AI/ML system implementation (embeddings, vector search)
- ✅ Production engineering (error handling, cost optimization, scalability)
- ✅ Data pipeline design (PDF processing, chunking, metadata extraction)
- ✅ API integration (Gemini, ChromaDB)
- ✅ Technical documentation and teaching
- ✅ Software engineering best practices

**Relevant for roles:**
- AI Engineer (Salesforce Agentforce, OpenAI, Anthropic)
- ML Engineer (RAG system implementation)
- Data Scientist (NLP, information retrieval)
- Research Engineer (academic tools development)

---

## 📖 Tutorial Overview

| Tutorial | Topic | Time | Key Takeaways |
|----------|-------|------|---------------|
| **01** | Embeddings | 40-50 min | Vector representations, cosine similarity |
| **02** | Chunking | 40-50 min | Document preprocessing, recursive splitting |
| **03** | Vector DB | 40-50 min | ChromaDB, semantic search, metadata |
| **04** | RAG Pipeline | 50-60 min | End-to-end system, prompt engineering |
| **DEMO** | Full System | 10 min | Everything integrated, production-ready |

**Total Learning Time:** 3-4 hours for complete mastery  
**Prerequisites:** Basic Python, curiosity about AI  
**Outcome:** Build production RAG systems for any domain  

---

## 💡 What Makes RAG Powerful?

### Without RAG (Baseline LLM):
```
Question: "What are BERT's key innovations?"
Answer: "BERT introduced bidirectional training and achieved 
         state-of-the-art results on many NLP tasks..."
         
Issues: ❌ Generic ❌ No sources ❌ Might be inaccurate
```

### With RAG (Context-Enhanced):
```
Question: "What are BERT's key innovations?"
Answer: "BERT's key innovations include bidirectional pre-training 
         using masked language modeling (MLM) and next sentence 
         prediction (NSP), allowing it to understand context from 
         both directions [Devlin et al., 2018, Section 3.1]"
         
Benefits: ✅ Specific ✅ Cited ✅ Verifiable ✅ Accurate
```

---

## 🔧 System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- Internet connection (for API calls)

**Recommended:**
- Python 3.10+
- 8 GB RAM
- GPU (optional - speeds up embedding generation)

**Tested On:**
- ✅ Google Colab (free tier)
- ✅ Local Jupyter notebooks
- ✅ MacOS, Linux, Windows

---

## 📞 Contact & Support

**Author:** Ram Prashanth Rao G  
**Email:** gajarghat.r@northeastern.edu  
**Institution:** Northeastern University  
**Course:** INFO 7390 - Art and Science of Data  

**Questions?** Open an issue or reach out via email!

**Found this helpful?** ⭐ Star the repository!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**TL;DR:** You can use this code for learning, teaching, or building your own projects. Attribution appreciated! 🙏

---

## 🙏 Acknowledgments

**Inspired by:**
- LangChain and LlamaIndex frameworks
- Original RAG paper by Lewis et al.
- The open-source AI community

**Built with:**
- Sentence Transformers by UKPLab
- ChromaDB by Chroma
- Google Gemini API
- Lots of coffee ☕ and curiosity 🚀

---

## 📊 Stats

![GitHub repo size](https://img.shields.io/github/repo-size/Ramprashanth17/Gen_AI)
![GitHub last commit](https://img.shields.io/github/last-commit/Ramprashanth17/Gen_AI)

**Project Timeline:** December 2025  
**Development Time:** ~40 hours  
**Lines of Code:** ~2,000  
**Tutorials Created:** 4 + 1 demo  
**Concepts Taught:** Embeddings, Chunking, Vector DBs, RAG, Prompt Engineering  

---

## 🌟 Final Thoughts

**RAG is transforming how we build AI applications.** Instead of hoping LLMs have the right knowledge, we give them exactly what they need, when they need it.

**This project is my contribution to making RAG accessible** - breaking down a complex system into understandable pieces that anyone can learn and build.

**Start your journey:** [Tutorial 01: Embeddings Fundamentals](notebooks/1_Embeddings_Fundamentals.ipynb)

---

*Built with 💙 for learners, by a learner.*

**Happy learning! 🚀**
