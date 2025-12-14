"""
Vector database utilities for face embeddings using pgvector.
"""
import numpy as np
from typing import Optional, List
import logging

try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    logging.warning("pgvector not available. Install with: pip install pgvector[sqlalchemy]")

from app.config import settings

logger = logging.getLogger(__name__)


class VectorDB:
    """Vector database interface for face embeddings."""
    
    @staticmethod
    def get_vector_type(embedding_dim: int = 512):
        """
        Get SQLAlchemy Vector type for embeddings.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            
        Returns:
            SQLAlchemy Vector column type
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector is not installed. Install with: pip install pgvector[sqlalchemy]"
            )
        
        return Vector(embedding_dim)
    
    @staticmethod
    def numpy_to_vector(embedding: np.ndarray) -> List[float]:
        """
        Convert numpy array to list for vector storage.
        
        Args:
            embedding: Numpy array embedding
            
        Returns:
            List of floats
        """
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
    
    @staticmethod
    def vector_to_numpy(vector: List[float]) -> np.ndarray:
        """
        Convert vector (list) to numpy array.
        
        Args:
            vector: List of floats or numpy array
            
        Returns:
            Numpy array
        """
        if isinstance(vector, np.ndarray):
            return vector
        return np.array(vector, dtype=np.float32)
    
    @staticmethod
    async def search_similar(
        db_session,
        model_class,
        query_embedding: np.ndarray,
        candidate_id: Optional[str] = None,
        limit: int = 1,
        threshold: float = 0.0
    ):
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            db_session: Database session
            model_class: SQLAlchemy model class with embedding column
            query_embedding: Query embedding vector
            candidate_id: Optional candidate_id filter
            limit: Maximum number of results
            threshold: Minimum similarity threshold
            
        Returns:
            List of matching records
        """
        from sqlalchemy import func, select
        
        # Convert to list for pgvector
        query_vector = VectorDB.numpy_to_vector(query_embedding)
        
        # Build query with cosine similarity
        query = select(
            model_class,
            (1 - func.cosine_distance(model_class.embedding, query_vector)).label('similarity')
        )
        
        if candidate_id:
            query = query.where(model_class.candidate_id == candidate_id)
        
        query = query.order_by(
            func.cosine_distance(model_class.embedding, query_vector)
        ).limit(limit)
        
        result = await db_session.execute(query)
        rows = result.all()
        
        # Filter by threshold
        filtered = [
            (row[0], row[1]) for row in rows 
            if row[1] >= threshold
        ]
        
        return filtered

