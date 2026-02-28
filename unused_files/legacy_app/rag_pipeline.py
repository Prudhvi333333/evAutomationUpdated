"""
RAG pipeline for question answering using Ollama and vector retrieval
"""

from typing import List, Dict
import ollama
import config
from vector_store import VectorStore


class RAGPipeline:
    """Handles RAG-based question answering"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.model = config.OLLAMA_MODEL
    
    def retrieve_context(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict]:
        """Retrieve relevant context from vector store"""
        results = self.vector_store.search(query, top_k=top_k)
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into context string"""
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('filename', 'Unknown')
            text = result['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """Generate prompt for the LLM"""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question based on the context provided above
- If the context doesn't contain enough information to answer the question, say so
- Be concise and accurate
- Cite the source when relevant

Answer:"""
        return prompt
    
    def query(self, question: str) -> Dict:
        """Process a question through the RAG pipeline"""
        # Retrieve relevant context
        results = self.retrieve_context(question)
        context = self.format_context(results)
        
        # Generate prompt
        prompt = self.generate_prompt(question, context)
        
        # Get response from Ollama
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': config.TEMPERATURE
                }
            )
            
            answer = response['response']
            
            return {
                'answer': answer,
                'context': results,
                'success': True
            }
        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'context': results,
                'success': False,
                'error': str(e)
            }
    
    def stream_query(self, question: str):
        """Process a question with streaming response"""
        # Retrieve relevant context
        results = self.retrieve_context(question)
        context = self.format_context(results)
        
        # Generate prompt
        prompt = self.generate_prompt(question, context)
        
        # Stream response from Ollama
        try:
            stream = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': config.TEMPERATURE
                }
            )
            
            for chunk in stream:
                yield chunk['response']
                
        except Exception as e:
            yield f"Error: {str(e)}"
