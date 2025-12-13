"""RAG System Evaluation"""

import sys
sys.path.append('..')

from src.rag_pipeline import RAGPipeline
from .test_queries import SAP_TEST_QUERIES, SALESFORCE_TEST_QUERIES
from dotenv import load_dotenv
import time
import json

load_dotenv()

def evaluate_collection(pipeline, test_queries, collection_name):
    """Evaluate RAG performance on a collection"""
    results = []
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {collection_name}")
    print(f"{'='*60}")
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Testing: {test['question'][:50]}...")
        
        start_time = time.time()
        result = pipeline.query(test['question'], collection_name=collection_name, top_k=5)
        elapsed = time.time() - start_time
        
        # Check if expected doc in top results
        top_sources = [s['source'] for s in result['sources']]
        hit = test['expected_doc'] in top_sources
        
        # Get top similarity score
        top_similarity = result['sources'][0]['similarity'] if result['sources'] else 0
        
        eval_result = {
            'question': test['question'],
            'expected_doc': test['expected_doc'],
            'retrieved_docs': top_sources,
            'hit': hit,
            'top_similarity': top_similarity,
            'response_time': elapsed,
            'answer_length': len(result['answer'])
        }
        
        results.append(eval_result)
        
        # Print result
        status = "✅ HIT" if hit else "❌ MISS"
        print(f"  {status} | Top sim: {top_similarity:.1%} | Time: {elapsed:.2f}s")
        if not hit:
            print(f"  Expected: {test['expected_doc']}")
            print(f"  Got: {top_sources[0]}")
    
    return results

def calculate_metrics(results):
    """Calculate aggregate metrics"""
    total = len(results)
    hits = sum(1 for r in results if r['hit'])
    
    metrics = {
        'total_queries': total,
        'retrieval_accuracy': hits / total,
        'avg_top_similarity': sum(r['top_similarity'] for r in results) / total,
        'avg_response_time': sum(r['response_time'] for r in results) / total,
        'avg_answer_length': sum(r['answer_length'] for r in results) / total
    }
    
    return metrics

def main():
    print("🧪 RAG SYSTEM EVALUATION")
    print("="*60)
    
    pipeline = RAGPipeline()
    
    # Evaluate SAP
    sap_results = evaluate_collection(pipeline, SAP_TEST_QUERIES, "sap_knowledge")
    sap_metrics = calculate_metrics(sap_results)
    
    # Evaluate Salesforce
    sf_results = evaluate_collection(pipeline, SALESFORCE_TEST_QUERIES, "salesforce_knowledge")
    sf_metrics = calculate_metrics(sf_results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    print("\n🔷 SAP Knowledge Base:")
    print(f"  Retrieval Accuracy: {sap_metrics['retrieval_accuracy']:.1%}")
    print(f"  Avg Top Similarity: {sap_metrics['avg_top_similarity']:.1%}")
    print(f"  Avg Response Time: {sap_metrics['avg_response_time']:.2f}s")
    
    print("\n🔶 Salesforce Knowledge Base:")
    print(f"  Retrieval Accuracy: {sf_metrics['retrieval_accuracy']:.1%}")
    print(f"  Avg Top Similarity: {sf_metrics['avg_top_similarity']:.1%}")
    print(f"  Avg Response Time: {sf_metrics['avg_response_time']:.2f}s")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'sap': {'results': sap_results, 'metrics': sap_metrics},
            'salesforce': {'results': sf_results, 'metrics': sf_metrics}
        }, f, indent=2)
    
    print(f"\n✅ Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()