from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()
pipeline = RAGPipeline()
print('✅ RAGPipeline imports!')

result = pipeline.query("What is SAP's AI policy?", top_k=3)
print(f'✅ Query works! Answer: {len(result["answer"])} chars')
print(f'✅ Sources: {len(result["sources"])}')
