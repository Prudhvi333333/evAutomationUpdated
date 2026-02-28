"""
Phase 3: Chunking Tests
Tests the chunking strategies and validates chunk quality
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

# Configuration
EXCEL_FILE = "GNEM updated excel (1).xlsx"
RESULTS_FILE = "test_results.json"


class TestChunker:
    """Test chunking strategies."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns.tolist()
        self.all_chunks = []
    
    def create_row_chunks(self) -> List[Dict]:
        """Create full row chunks."""
        chunks = []
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
        return chunks
    
    def create_semantic_chunks(self) -> List[Dict]:
        """Create semantic field-grouped chunks."""
        semantic_groups = {
            'company_identity': ['Company', 'Category', 'Industry Group'],
            'location_facility': ['Location', 'Primary Facility Type'],
            'supply_chain': ['EV Supply Chain Role', 'Primary OEMs'],
            'product_details': ['Product / Service', 'EV / Battery Relevant']
        }
        
        chunks = []
        for idx, row in self.df.iterrows():
            for group_name, group_cols in semantic_groups.items():
                fields = []
                for col in group_cols:
                    if col in self.columns:
                        value = row[col]
                        if pd.notna(value) and str(value).strip():
                            fields.append(f"{col}: {value}")
                
                if fields:
                    chunks.append({
                        'text': f"[{group_name}] " + " | ".join(fields),
                        'metadata': {
                            'row_index': int(idx),
                            'company': str(row.get('Company', 'Unknown')),
                            'semantic_group': group_name,
                            'chunk_type': 'semantic'
                        }
                    })
        return chunks
    
    def create_keyword_chunks(self) -> List[Dict]:
        """Create keyword-indexed chunks."""
        chunks = []
        for idx, row in self.df.iterrows():
            company = row.get('Company', '')
            if pd.notna(company) and str(company).strip():
                fields = []
                for col in ['Category', 'EV Supply Chain Role', 'Primary OEMs', 'Location']:
                    if col in self.columns:
                        val = row[col]
                        if pd.notna(val):
                            fields.append(f"{col}: {val}")
                
                chunks.append({
                    'text': f"Company: {company} | " + " | ".join(fields),
                    'metadata': {
                        'row_index': int(idx),
                        'company': str(company),
                        'search_key': str(company).lower(),
                        'chunk_type': 'keyword_indexed'
                    }
                })
        return chunks


def test_row_chunking():
    """Test full row chunking."""
    print("\n" + "="*80)
    print("TEST 1: Row Chunking Strategy")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_row_chunks()
        
        print(f"✓ Created {len(chunks)} row chunks")
        print(f"  Expected: {len(df)} chunks")
        
        # Verify all rows covered
        row_indices = {c['metadata']['row_index'] for c in chunks}
        all_rows_covered = row_indices == set(range(len(df)))
        
        # Check chunk sizes
        sizes = [len(c['text']) for c in chunks]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        print(f"  All rows covered: {all_rows_covered}")
        print(f"  Average chunk size: {avg_size:.0f} chars")
        print(f"  Sample chunk:\n    {chunks[0]['text'][:150]}...")
        
        return {
            'test': 'row_chunking',
            'status': 'PASS' if all_rows_covered else 'FAIL',
            'chunks_created': len(chunks),
            'expected_chunks': len(df),
            'all_rows_covered': all_rows_covered,
            'avg_chunk_size': avg_size,
            'sample_chunk': chunks[0]['text'][:200] if chunks else None,
            'message': 'Row chunking validated' if all_rows_covered else 'Not all rows covered'
        }
    except Exception as e:
        print(f"✗ Row chunking failed: {e}")
        return {
            'test': 'row_chunking',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Row chunking failed'
        }


def test_semantic_chunking():
    """Test semantic chunking."""
    print("\n" + "="*80)
    print("TEST 2: Semantic Chunking Strategy")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_semantic_chunks()
        
        print(f"✓ Created {len(chunks)} semantic chunks")
        
        # Check group distribution
        group_counts = defaultdict(int)
        for c in chunks:
            group_counts[c['metadata']['semantic_group']] += 1
        
        print(f"  Semantic groups: {dict(group_counts)}")
        
        # Verify groups have content
        empty_groups = [g for g, count in group_counts.items() if count == 0]
        
        sample = chunks[0] if chunks else None
        if sample:
            print(f"  Sample chunk:\n    {sample['text'][:150]}...")
        
        return {
            'test': 'semantic_chunking',
            'status': 'PASS' if not empty_groups else 'FAIL',
            'chunks_created': len(chunks),
            'group_distribution': dict(group_counts),
            'empty_groups': empty_groups,
            'sample_chunk': sample['text'][:200] if sample else None,
            'message': 'Semantic chunking validated' if not empty_groups else 'Some groups empty'
        }
    except Exception as e:
        print(f"✗ Semantic chunking failed: {e}")
        return {
            'test': 'semantic_chunking',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Semantic chunking failed'
        }


def test_keyword_chunking():
    """Test keyword-indexed chunking."""
    print("\n" + "="*80)
    print("TEST 3: Keyword-Indexed Chunking")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_keyword_chunks()
        
        print(f"✓ Created {len(chunks)} keyword chunks")
        
        # Check that all companies are represented
        companies_with_chunks = {c['metadata']['company'] for c in chunks}
        all_companies = set(df['Company'].dropna().astype(str))
        coverage = len(companies_with_chunks) / len(all_companies) if all_companies else 0
        
        print(f"  Company coverage: {len(companies_with_chunks)}/{len(all_companies)} ({coverage*100:.1f}%)")
        
        sample = chunks[0] if chunks else None
        if sample:
            print(f"  Sample chunk:\n    {sample['text'][:150]}...")
        
        return {
            'test': 'keyword_chunking',
            'status': 'PASS' if coverage >= 0.9 else 'WARN',
            'chunks_created': len(chunks),
            'companies_covered': len(companies_with_chunks),
            'total_companies': len(all_companies),
            'coverage_percent': round(coverage * 100, 1),
            'sample_chunk': sample['text'][:200] if sample else None,
            'message': f'Keyword chunking: {coverage*100:.1f}% coverage'
        }
    except Exception as e:
        print(f"✗ Keyword chunking failed: {e}")
        return {
            'test': 'keyword_chunking',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Keyword chunking failed'
        }


def test_chunk_quality():
    """Test overall chunk quality metrics."""
    print("\n" + "="*80)
    print("TEST 4: Chunk Quality Metrics")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        
        # Generate all chunks
        row_chunks = chunker.create_row_chunks()
        semantic_chunks = chunker.create_semantic_chunks()
        keyword_chunks = chunker.create_keyword_chunks()
        
        all_chunks = row_chunks + semantic_chunks + keyword_chunks
        
        # Size analysis
        sizes = [len(c['text']) for c in all_chunks]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        
        print(f"✓ Total chunks: {len(all_chunks)}")
        print(f"  Size range: {min_size} - {max_size} chars")
        print(f"  Average: {avg_size:.0f} chars")
        
        # Check for problematic chunks
        very_small = sum(1 for s in sizes if s < 50)
        very_large = sum(1 for s in sizes if s > 2000)
        
        print(f"  Very small chunks (<50 chars): {very_small}")
        print(f"  Very large chunks (>2000 chars): {very_large}")
        
        # Field coverage
        field_coverage = defaultdict(int)
        for chunk in all_chunks:
            for col in df.columns:
                if col in chunk['text']:
                    field_coverage[col] += 1
        
        print(f"  All fields covered: {len(field_coverage)}/{len(df.columns)}")
        
        quality_score = 'PASS'
        if very_small > len(all_chunks) * 0.1:
            quality_score = 'WARN'
        if very_large > len(all_chunks) * 0.1:
            quality_score = 'WARN'
        
        return {
            'test': 'chunk_quality',
            'status': quality_score,
            'total_chunks': len(all_chunks),
            'chunks_by_type': {
                'full_row': len(row_chunks),
                'semantic': len(semantic_chunks),
                'keyword': len(keyword_chunks)
            },
            'size_stats': {
                'min': min_size,
                'max': max_size,
                'avg': round(avg_size, 1)
            },
            'problematic_chunks': {
                'very_small': very_small,
                'very_large': very_large
            },
            'field_coverage': dict(field_coverage),
            'message': f'Chunk quality: avg {avg_size:.0f} chars, {len(field_coverage)} fields covered'
        }
    except Exception as e:
        print(f"✗ Chunk quality test failed: {e}")
        return {
            'test': 'chunk_quality',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Chunk quality test failed'
        }


def test_embedding_readiness():
    """Test that chunks are ready for embedding."""
    print("\n" + "="*80)
    print("TEST 5: Embedding Readiness")
    print("="*80)
    
    try:
        df = pd.read_excel(EXCEL_FILE)
        chunker = TestChunker(df)
        chunks = chunker.create_row_chunks() + chunker.create_semantic_chunks()
        
        # Check metadata completeness
        required_metadata = ['row_index', 'company', 'chunk_type']
        
        complete_metadata = 0
        for chunk in chunks:
            if all(key in chunk['metadata'] for key in required_metadata):
                complete_metadata += 1
        
        metadata_coverage = complete_metadata / len(chunks) if chunks else 0
        
        print(f"✓ Chunks with complete metadata: {complete_metadata}/{len(chunks)}")
        print(f"  Metadata coverage: {metadata_coverage*100:.1f}%")
        
        # Check text content
        empty_text = sum(1 for c in chunks if not c['text'].strip())
        
        print(f"  Chunks with empty text: {empty_text}")
        
        ready = metadata_coverage >= 0.95 and empty_text == 0
        
        return {
            'test': 'embedding_readiness',
            'status': 'PASS' if ready else 'FAIL',
            'total_chunks': len(chunks),
            'complete_metadata': complete_metadata,
            'metadata_coverage': round(metadata_coverage * 100, 1),
            'empty_text_chunks': empty_text,
            'ready_for_embedding': ready,
            'message': 'Chunks ready for embedding' if ready else 'Chunks need cleanup'
        }
    except Exception as e:
        print(f"✗ Embedding readiness test failed: {e}")
        return {
            'test': 'embedding_readiness',
            'status': 'FAIL',
            'error': str(e),
            'message': 'Embedding readiness test failed'
        }


def run_all_tests():
    """Run all Phase 3 tests."""
    print("\n" + "="*80)
    print("PHASE 3: CHUNKING TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'phase': 'phase_3_chunking',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }
    
    # Run tests
    results['tests'].append(test_row_chunking())
    results['tests'].append(test_semantic_chunking())
    results['tests'].append(test_keyword_chunking())
    results['tests'].append(test_chunk_quality())
    results['tests'].append(test_embedding_readiness())
    
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
    print("PHASE 3 SUMMARY")
    print("="*80)
    print(f"Total tests: {len(results['tests'])}")
    print(f"Passed: {passed} ✓")
    print(f"Warnings: {warnings} ⚠")
    print(f"Failed: {failed} ✗")
    print(f"\nCan proceed to Phase 4: {results['summary']['can_proceed']}")
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    return results


if __name__ == "__main__":
    # Change to project root to find Excel file
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    results = run_all_tests()
    
    if results['summary']['can_proceed']:
        print("\n✓ Phase 3 complete - Ready for Phase 4")
        sys.exit(0)
    else:
        print("\n✗ Phase 3 incomplete - Fix issues before proceeding")
        sys.exit(1)
