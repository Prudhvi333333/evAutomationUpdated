"""
Phase 5: Full Pipeline Integration
Runs complete RAG pipeline with model responses and generates Excel output
"""

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict
from docx import Document

# Load existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

# Try to import optional dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
EXCEL_FILE = "GNEM updated excel (1).xlsx"
QUESTIONS_FILE = "Sample questions.xlsx"
OUTPUT_EXCEL = "test_results_all_101.xlsx"
NUM_QUESTIONS = 101

# Get working model from Phase 1 results
def get_working_models():
    """Load working models from Phase 1 results."""
    try:
        with open(os.path.join('pipeline', 'phase_1_model_check', 'test_results.json'), 'r') as f:
            phase1_results = json.load(f)
            
        tinyllama_working = False
        gemini_model = None
        
        for test in phase1_results.get('tests', []):
            if test['test'] == 'tinyllama_model' and test['status'] == 'PASS':
                tinyllama_working = True
            if test['test'] == 'gemini_models' and test['status'] == 'PASS':
                gemini_model = test.get('recommended_model', 'gemini-1.5-flash')
        
        return tinyllama_working, gemini_model
    except:
        # Defaults if Phase 1 not run
        return True, 'models/gemini-2.5-flash'


class PipelineChunker:
    """Production chunker for the full pipeline."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns.tolist()
    
    def create_all_chunks(self) -> List[Dict]:
        """Create comprehensive chunk set."""
        chunks = []
        
        # 1. Full row chunks
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
        
        # 2. Keyword chunks for company lookups
        for idx, row in self.df.iterrows():
            company = row.get('Company', '')
            if pd.notna(company) and str(company).strip():
                fields = [f"Company: {company}"]
                for col in ['Category', 'EV Supply Chain Role', 'Primary OEMs', 'Location', 'Product / Service']:
                    if col in self.columns:
                        val = row[col]
                        if pd.notna(val):
                            fields.append(f"{col}: {val}")
                
                chunks.append({
                    'text': " | ".join(fields),
                    'metadata': {
                        'row_index': int(idx),
                        'company': str(company),
                        'chunk_type': 'company_profile'
                    }
                })
        
        return chunks


class PipelineRetriever:
    """Production retriever for the full pipeline."""
    
    def __init__(self, vector_store: VectorStore, df: pd.DataFrame):
        self.vector_store = vector_store
        self.df = df
    
    def retrieve(self, question: str, top_k: int = 8) -> List[Dict]:
        """Multi-strategy retrieval."""
        # Vector search
        results = self.vector_store.search(question, top_k=top_k)
        
        # Boost by company name match
        for r in results:
            company = r['metadata'].get('company', '')
            if company and company.lower() in question.lower():
                r['score'] = r.get('score', 0) * 1.5
        
        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return results[:top_k]


def extract_questions(filepath: str, limit: int = 101) -> List[str]:
    """Extract questions from Excel file."""
    df = pd.read_excel(filepath)
    # Assuming questions are in the first column
    questions = df.iloc[:, 0].dropna().tolist()
    
    print(f"  [DEBUG] Total questions in Excel: {len(questions)}")
    
    # Clean and validate
    valid_questions = []
    for q in questions:
        if isinstance(q, str) and len(q.strip()) > 10:
            valid_questions.append(q.strip())
    
    print(f"  [DEBUG] Valid questions: {len(valid_questions)}")
    
    if len(valid_questions) < limit:
        print(f"  [WARNING] Only {len(valid_questions)} questions available, but {limit} requested")
        limit = len(valid_questions)
    
    return valid_questions[:limit]


def query_tinyllama_rag(retriever: PipelineRetriever, question: str) -> Dict:
    """Query TinyLlama with RAG."""
    import time
    start = time.time()
    
    try:
        results = retriever.retrieve(question, top_k=8)
        
        # Format context
        context_parts = []
        for i, r in enumerate(results, 1):
            company = r['metadata'].get('company', 'Unknown')
            context_parts.append(f"[Source {i}: {company}]\n{r['text']}")
        context = "\n\n".join(context_parts) if context_parts else "No relevant context."
        
        prompt = f"""Answer based ONLY on the provided context about Georgia EV supply chain companies.

Context:
{context}

Question: {question}

Provide a concise, specific answer citing relevant companies.

Answer:"""
        
        response = ollama.generate(
            model="qwen2.5:14b",
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 300}
        )
        
        return {
            'answer': response['response'],
            'success': True,
            'time': round(time.time() - start, 2)
        }
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'success': False,
            'time': round(time.time() - start, 2)
        }


def query_gemini_rag(retriever: PipelineRetriever, question: str, model: str) -> Dict:
    """Query Gemini with RAG."""
    import time
    start = time.time()
    
    try:
        results = retriever.retrieve(question, top_k=8)
        
        # Format context
        context_parts = []
        for i, r in enumerate(results, 1):
            company = r['metadata'].get('company', 'Unknown')
            context_parts.append(f"[Source {i}: {company}]\n{r['text']}")
        context = "\n\n".join(context_parts) if context_parts else "No relevant context."
        
        prompt = f"""Answer based ONLY on the provided context about Georgia EV supply chain companies.

Context:
{context}

Question: {question}

Provide a concise, specific answer citing relevant companies. If the answer is not in the context, say "Not found in data".

Answer:"""
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'success': True,
            'time': round(time.time() - start, 2)
        }
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'success': False,
            'time': round(time.time() - start, 2)
        }


def query_gemini_no_rag(question: str, model: str) -> Dict:
    """Query Gemini without RAG."""
    import time
    start = time.time()
    
    try:
        prompt = f"""Answer the following question to the best of your knowledge. Be concise.

Question: {question}

Answer:"""
        
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        gemini_model = genai.GenerativeModel(model)
        response = gemini_model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'success': True,
            'time': round(time.time() - start, 2)
        }
    except Exception as e:
        return {
            'answer': f"Error: {str(e)}",
            'success': False,
            'time': round(time.time() - start, 2)
        }


def run_full_pipeline():
    """Run the complete pipeline."""
    print("\n" + "="*80)
    print("PHASE 5: FULL PIPELINE INTEGRATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check prerequisites
    tinyllama_working, gemini_model = get_working_models()
    
    print(f"\nModels configured:")
    print(f"  TinyLlama: {'✓' if tinyllama_working else '✗ (will skip)'}")
    print(f"  Gemini: {gemini_model if gemini_model else '✗ (will skip)'}")
    
    if not tinyllama_working and not gemini_model:
        print("\n✗ No working models found! Run Phase 1 first.")
        return None
    
    # Load data
    print(f"\nLoading data from {EXCEL_FILE}...")
    df = pd.read_excel(EXCEL_FILE)
    print(f"  Loaded {len(df)} rows")
    
    # Setup vector store
    print("\nSetting up vector store...")
    vector_store = VectorStore()
    try:
        vector_store.delete_collection()
    except:
        pass
    
    chunker = PipelineChunker(df)
    chunks = chunker.create_all_chunks()
    print(f"  Created {len(chunks)} chunks")
    
    added = vector_store.add_documents(chunks)
    print(f"  Added {added} documents to vector store")
    
    retriever = PipelineRetriever(vector_store, df)
    
    # Extract questions
    print(f"\nExtracting {NUM_QUESTIONS} questions from {QUESTIONS_FILE}...")
    questions = extract_questions(QUESTIONS_FILE, NUM_QUESTIONS)
    print(f"  Found {len(questions)} questions")
    
    if not questions:
        print("✗ No questions found!")
        return None
    
    # Run tests
    print(f"\nRunning tests on {len(questions)} questions...")
    results = []
    
    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}/{len(questions)}: {question[:80]}...")
        print(f"{'='*80}")
        
        row = {
            'question_number': idx,
            'question': question
        }
        
        # Test 1: TinyLlama with RAG
        if tinyllama_working:
            print("  [1/3] TinyLlama RAG...")
            result = query_tinyllama_rag(retriever, question)
            row['tinyllama_rag'] = result['answer']
            row['tinyllama_rag_time'] = result['time']
            row['tinyllama_rag_success'] = result['success']
        else:
            row['tinyllama_rag'] = "Model not available"
            row['tinyllama_rag_time'] = 0
            row['tinyllama_rag_success'] = False
        
        # Test 2: Gemini with RAG
        if gemini_model:
            print("  [2/3] Gemini RAG...")
            result = query_gemini_rag(retriever, question, gemini_model)
            row['gemini_rag'] = result['answer']
            row['gemini_rag_time'] = result['time']
            row['gemini_rag_success'] = result['success']
        else:
            row['gemini_rag'] = "Model not available"
            row['gemini_rag_time'] = 0
            row['gemini_rag_success'] = False
        
        # Test 3: Gemini without RAG
        if gemini_model:
            print("  [3/3] Gemini No RAG...")
            result = query_gemini_no_rag(question, gemini_model)
            row['gemini_no_rag'] = result['answer']
            row['gemini_no_rag_time'] = result['time']
            row['gemini_no_rag_success'] = result['success']
        else:
            row['gemini_no_rag'] = "Model not available"
            row['gemini_no_rag_time'] = 0
            row['gemini_no_rag_success'] = False
        
        results.append(row)
        print(f"  Completed {idx}/{len(questions)}")
        
        # Rate limiting: wait 5s between questions to avoid Gemini quota
        if gemini_model and idx < len(questions):
            print(f"  [Rate limit] Waiting 5s before next question...")
            time.sleep(5)
    
    # Generate Excel
    print(f"\n\nGenerating Excel report...")
    
    # Create dataframes
    rows = []
    for r in results:
        rows.append({
            'Question Number': r['question_number'],
            'Question': r['question'],
            'TinyLlama RAG': r['tinyllama_rag'],
            'TinyLlama Time (s)': r['tinyllama_rag_time'],
            'TinyLlama Success': r['tinyllama_rag_success'],
            'Gemini RAG': r['gemini_rag'],
            'Gemini RAG Time (s)': r['gemini_rag_time'],
            'Gemini RAG Success': r['gemini_rag_success'],
            'Gemini No RAG': r['gemini_no_rag'],
            'Gemini No RAG Time (s)': r['gemini_no_rag_time'],
            'Gemini No RAG Success': r['gemini_no_rag_success']
        })
    
    df_results = pd.DataFrame(rows)
    
    # Create summary
    tiny_success = sum(1 for r in results if r['tinyllama_rag_success'])
    gemini_rag_success = sum(1 for r in results if r['gemini_rag_success'])
    gemini_no_rag_success = sum(1 for r in results if r['gemini_no_rag_success'])
    
    # Write to Excel
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='Model Responses', index=False)
        
        # Adjust column widths
        worksheet = writer.sheets['Model Responses']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if cell.value and len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 100)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"\n✓ Results saved to: {OUTPUT_EXCEL}")
    print(f"\nSuccess Summary:")
    if tinyllama_working:
        print(f"  TinyLlama RAG: {tiny_success}/{len(results)}")
    if gemini_model:
        print(f"  Gemini RAG: {gemini_rag_success}/{len(results)}")
        print(f"  Gemini No RAG: {gemini_no_rag_success}/{len(results)}")
    
    # Save test results JSON
    test_results = {
        'phase': 'phase_5_full_pipeline',
        'timestamp': datetime.now().isoformat(),
        'models_used': {
            'tinyllama': tinyllama_working,
            'gemini': gemini_model
        },
        'questions_tested': len(results),
        'summary': {
            'tinyllama_success': tiny_success,
            'gemini_rag_success': gemini_rag_success,
            'gemini_no_rag_success': gemini_no_rag_success
        }
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n" + "="*80)
    print("PHASE 5 COMPLETE")
    print("="*80)
    
    return test_results


if __name__ == "__main__":
    # Change to project root
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    results = run_full_pipeline()
    
    if results:
        print("\n✓ Full pipeline completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Full pipeline failed")
        sys.exit(1)
