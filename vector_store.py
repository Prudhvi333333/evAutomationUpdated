"""
Vector store operations using Qdrant
"""

from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import config
import uuid


class VectorStore:
    """Manages vector storage operations with Qdrant"""
    
    def __init__(self):
        # Initialize Qdrant client
        if config.QDRANT_USE_MEMORY:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT
            )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Create collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if config.COLLECTION_NAME not in collection_names:
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=config.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 50) -> int:
        """Add document chunks to vector store in batches"""
        total_added = 0
        
        # Process in batches to manage memory
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            points = []
            
            for chunk in batch:
                # Generate embedding
                embedding = self.generate_embedding(chunk['text'])
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'text': chunk['text'],
                        'metadata': chunk.get('metadata', {})
                    }
                )
                points.append(point)
            
            # Upload batch to Qdrant
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            total_added += len(points)
        
        return total_added
    
    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in Qdrant
        try:
            search_results = self.client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=query_embedding,
                limit=top_k
            ).points
        except AttributeError:
            search_results = self.client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k
            )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                'text': result.payload['text'],
                'metadata': result.payload.get('metadata', {}),
                'score': result.score
            })
        
        return results
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(config.COLLECTION_NAME)
            return {
                'name': config.COLLECTION_NAME,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count
            }
        except Exception as e:
            return {
                'name': config.COLLECTION_NAME,
                'error': str(e)
            }
    
    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(config.COLLECTION_NAME)
        self._initialize_collection()
