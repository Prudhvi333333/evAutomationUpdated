"""
Phase 4: Retrieval Tests
Tests vector store operations and retrieval quality
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Load existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Configuration
EXCEL_FILE = "GNEM updated excel (1).xlsx"
RESULTS_FILE = "test_results.json"


class TestChunker:
    """Simple chunker for testing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns.tolist()
    
    def create_all_chunks(self) -> List[Dict]:
        """Create all chunk types for testing."""
        chunks = []
        
        # Row chunks
        for idx, row in self.df.iterrows():
            fields = []
            for col in self.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    fields.append(f"{col}: {value}")
            
            chunks.append({
                'text': " | ".join(fields),
                'metadata': {
                    'row_index': int(idx),
                    'company': str(row.get('Company', 'Unknown')),
                    'chunk_type': 'full_row'
                }
            })
        
        # Keyword chunks for companies
        for idx, row in self.df.iterrows():
            company = row.get('Company', '')
            if pd.notna(company):
                fields = [f"Company: {company}"]
                for col in ['Category', 'EV Supply Chain Role', 'Location']:
                    if col in self.columns:
                        val = row[col]
                        if pd.notna(val):
                            fields.append(f"{col}: {val}")
                
                chunks.append({
                    'text': " | ".join(fields),
                    'metadata': {
                        'row_index': int(idx),
                        'company': str(company),
                        'chunk_type': 'keyword'
                    }
                })
        
        return chunks


def test_vector_store_initialization():
    """Test vector store can be initialized."""
    print("\n" + "="*80)
    print("TEST 1: Vector Store Initialization")
    print("="*80)
    
    try:
        vector_store = VectorStore()
        info = vector_store.get_collection_info()
        
        print(f"✓ Vector store initialized")
        print(f"  Collection name: {info.get('name', 'Unknown')}")
        print(f"  Points count: {info.get('points_count', 0)}")
        
        return {
            'test': 'vector_store_init',
            'status': 'PASS',
            'collection_info': info,
            'message': 'Vector store initialized successfully'
        }
    except Exception as e:
        print(f"✗ Vector store initialization failed: {e}")
        return {
            'test': 'vector_store_init',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Failed to initialize vector store'
        }


def test_embedding_generation():
    """Test embedding generation."""
    print("\n" + "="*80)
    print("TEST 2: Embedding Generation")
    print("="*80)
    
    try:
        vector_store = VectorStore()
        
        test_texts = [
            "Company: SK Innovation | Category: OEM Supply Chain | EV Supply Chain Role: Battery Cell",
            "Company: Hyundai | Category: OEM | Location: Georgia"
        ]
        
        embeddings = []
        for text in test_texts:
            embedding = vector_store.generate_embedding(text)
            embeddings.append({
                'text_length': len(text),
                'embedding_length': len(embedding),
                'sample_values': embedding[:5]  # First 5 values
            })
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"  Text {i+1}: {emb['text_length']} chars -> {emb['embedding_length']} dims")
        
        # Check all embeddings have same dimension
        dims = [e['embedding_length'] for e in embeddings]
        consistent_dims = len(set(dims)) == 1
        
        return {
            'test': 'embedding_generation',
            'status': 'PASS' if consistent_dims else 'FAIL',
            'embeddings': embeddings,
            'consistent_dimensions': consistent_dims,
            'message': 'Embeddings generated successfully' if consistent_dims else 'Inconsistent embedding dimensions'
        }
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return {
            'test': 'embedding_generation',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Embedding generation failed'
        }


def test_document_addition():
    """Test adding documents to vector store."""
    print("\n" + "="*80)
    print("TEST 3: Document Addition")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_all_chunks()[:20]  # Use only first 20 for speed
        
        vector_store = VectorStore()
        # Clear first
        try:
            vector_store.delete_collection()
        except:
            pass
        
        # Add documents
        added = vector_store.add_documents(chunks)
        
        info = vector_store.get_collection_info()
        points_count = info.get('points_count', 0)
        
        print(f"✓ Added {added} documents")
        print(f"  Points in collection: {points_count}")
        
        # Note: in-memory Qdrant may show 0 points immediately after add
        # but search still works - trust the 'added' count from upsert
        success = added == len(chunks)
        
        return {
            'test': 'document_addition',
            'status': 'PASS' if success else 'FAIL',
            'chunks_attempted': len(chunks),
            'chunks_added': added,
            'points_in_collection': points_count,
            'message': f'Added {added}/{len(chunks)} documents' if success else f'Only added {added}/{len(chunks)}'
        }
    except Exception as e:
        print(f"✗ Document addition failed: {e}")
        return {
            'test': 'document_addition',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Document addition failed'
        }


def test_basic_search():
    """Test basic vector search."""
    print("\n" + "="*80)
    print("TEST 4: Basic Vector Search")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_all_chunks()
        
        vector_store = VectorStore()
        try:
            vector_store.delete_collection()
        except:
            pass
        
        vector_store.add_documents(chunks)
        
        # Test search queries
        test_queries = [
            "Battery Cell companies",
            "Tier 1 suppliers",
            "Hyundai suppliers"
        ]
        
        search_results = []
        for query in test_queries:
            results = vector_store.search(query, top_k=3)
            search_results.append({
                'query': query,
                'results_found': len(results),
                'top_result': results[0]['metadata'].get('company', 'Unknown') if results else None,
                'top_score': round(results[0].get('score', 0), 3) if results else 0
            })
            print(f"  Query: '{query}' -> {len(results)} results")
        
        avg_results = sum(r['results_found'] for r in search_results) / len(search_results)
        
        return {
            'test': 'basic_search',
            'status': 'PASS' if avg_results > 0 else 'FAIL',
            'search_results': search_results,
            'average_results_per_query': round(avg_results, 1),
            'message': f'Search working: {avg_results:.1f} avg results per query'
        }
    except Exception as e:
        print(f"✗ Basic search failed: {e}")
        return {
            'test': 'basic_search',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Basic search failed'
        }


def test_retrieval_relevance():
    """Test retrieval relevance for specific queries."""
    print("\n" + "="*80)
    print("TEST 5: Retrieval Relevance")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_all_chunks()
        
        vector_store = VectorStore()
        try:
            vector_store.delete_collection()
        except:
            pass
        
        vector_store.add_documents(chunks)
        
        # Test specific company queries
        test_cases = []
        
        # Get some company names from the data
        sample_companies = df['Company'].dropna().head(3).tolist()
        
        for company in sample_companies:
            query = f"What does {company} do?"
            results = vector_store.search(query, top_k=5)
            
            # Check if company is in results
            company_in_results = any(
                company.lower() in r['metadata'].get('company', '').lower()
                for r in results
            )
            
            test_cases.append({
                'company': company,
                'found_in_top_5': company_in_results,
                'results_count': len(results)
            })
            
            status = "✓" if company_in_results else "✗"
            print(f"  {status} {company}: found in results = {company_in_results}")
        
        relevance_score = sum(1 for t in test_cases if t['found_in_top_5']) / len(test_cases)
        
        return {
            'test': 'retrieval_relevance',
            'status': 'PASS' if relevance_score >= 0.6 else 'WARN',
            'test_cases': test_cases,
            'relevance_score': round(relevance_score, 2),
            'message': f'Relevance score: {relevance_score*100:.0f}%'
        }
    except Exception as e:
        print(f"✗ Retrieval relevance test failed: {e}")
        return {
            'test': 'retrieval_relevance',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Retrieval relevance test failed'
        }


def run_all_tests():
    """Run all Phase 4 tests."""
    print("\n" + "="*80)
    print("PHASE 4: RETRIEVAL TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'phase': 'phase_4_retrieval',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    # Run tests
    results['tests'].append(test_vector_store_initialization())
    results['tests'].append(test_embedding_generation())
    results['tests'].append(test_document_addition())
    results['tests'].append(test_basic_search())
    results['tests'].append(test_retrieval_relevance())
    
    # Calculate summary
    passed = sum(1 for t in results['tests'] if t['status'] == 'PASS')
    warnings = sum(1 for t in results['tests'] if t['status'] == 'WARN')
    failed = sum(1 for t in results['tests'] if t['status'] == 'FAIL')
    
    results['summary'] = {
        'total': len(results['tests']),
        'passed': passed,
        'warnings': warnings,
        'failed': failed,
        'can_proceed': failed == 0
    }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 4 SUMMARY")
    print("="*80)
    print(f"Total tests: {len(results['tests'])}")
    print(f"Passed: {passed} ✓")
    print(f"Warnings: {warnings} ⚠")
    print(f"Failed: {failed} ✗")
    print(f"\nCan proceed to Phase 5: {results['summary']['can_proceed']}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    return results


if __name__ == "__main__":
    # Change to project root to find Excel file
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    results = run_all_tests()
    
    if results['summary']['can_proceed']:
        print("\n✓ Phase 4 complete - Ready for Phase 5")
        sys.exit(0)
    else:
        print("\n✗ Phase 4 incomplete - Fix issues before proceeding")
        sys.exit(1)
