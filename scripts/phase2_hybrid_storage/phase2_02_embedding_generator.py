# phase2_hybrid_storage/02_embedding_generator.py
import os
import numpy as np
from typing import List, Dict, Optional, Any
import logging
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Embedding Generation Engine
    Handles different embedding models with fallback strategies
    """
    
    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        """Initialize embedding generator with fallback strategies"""
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("D:\\m365\\models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.embedding_dimension = None
        self.model_name = None
        self.model_type = None  # 'transformer', 'tfidf', 'custom'
        
        # Load the best available model
        self.model = self._load_best_model(model_name)
        
        logger.info(f"✅ EmbeddingGenerator ready: {self.model_name} ({self.embedding_dimension}D)")
    
    def _load_best_model(self, preferred_model: Optional[str] = None):
        """Load the best available embedding model with multiple fallback strategies"""
        
        # Define model options in order of preference
        model_options = [
            {
                'name': preferred_model or 'all-MiniLM-L6-v2',
                'dimension': 384,
                'description': 'Fast and lightweight sentence transformer'
            },
            {
                'name': 'paraphrase-MiniLM-L6-v2',
                'dimension': 384,
                'description': 'Alternative lightweight model'
            },
            {
                'name': 'all-MiniLM-L12-v2',
                'dimension': 384,
                'description': 'Slightly larger model with better accuracy'
            },
            {
                'name': 'all-mpnet-base-v2',
                'dimension': 768,
                'description': 'High quality but slower'
            }
        ]
        
        logger.info("🧠 Loading embedding model...")
        
        # Strategy 1: Try Sentence Transformers
        for model_info in model_options:
            model_name = model_info['name']
            
            try:
                logger.info(f"  🔄 Trying {model_name}...")
                
                # Try cache-first loading
                model = self._try_load_transformer(model_name, model_info['dimension'])
                if model:
                    return model
                    
            except Exception as e:
                logger.debug(f"  ❌ {model_name} failed: {e}")
                continue
        
        # Strategy 2: Fallback to TF-IDF + SVD
        logger.warning("⚠️ All transformer models failed, using TF-IDF fallback...")
        return self._create_tfidf_model()
    
    def _try_load_transformer(self, model_name: str, dimension: int) -> Optional[SentenceTransformer]:
        """Try to load a sentence transformer model"""
        
        try:
            # First try cache-only
            model = SentenceTransformer(model_name, local_files_only=True)
            self.model_name = model_name
            self.embedding_dimension = dimension
            self.model_type = 'transformer'
            logger.info(f"  ✅ Loaded {model_name} from cache")
            return model
            
        except Exception:
            # Try online download
            try:
                # Clear any problematic environment variables
                os.environ.pop('HUGGINGFACE_HUB_TOKEN', None)
                
                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=False,
                    local_files_only=False,
                    cache_folder=str(self.cache_dir)
                )
                
                self.model_name = model_name
                self.embedding_dimension = dimension
                self.model_type = 'transformer'
                logger.info(f"  ✅ Downloaded and loaded {model_name}")
                return model
                
            except Exception as e:
                logger.debug(f"  ❌ Download failed for {model_name}: {e}")
                return None
    
    def _create_tfidf_model(self):
        """Create TF-IDF + SVD fallback model"""
        
        class TFIDFEmbeddingModel:
            def __init__(self, dimension: int = 384):
                self.dimension = dimension
                self.vectorizer = TfidfVectorizer(
                    max_features=2000,
                    stop_words='english',
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    max_df=0.8,
                    min_df=2
                )
                self.svd = TruncatedSVD(n_components=dimension, random_state=42)
                self.is_fitted = False
                self._corpus_cache = None
            
            def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
                if not isinstance(texts, list):
                    texts = [str(texts)]
                
                if not self.is_fitted:
                    logger.info("  🔧 Fitting TF-IDF model on input texts...")
                    tfidf_matrix = self.vectorizer.fit_transform(texts)
                    embeddings = self.svd.fit_transform(tfidf_matrix)
                    self.is_fitted = True
                    self._corpus_cache = texts.copy()  # Cache for future transforms
                else:
                    # For new texts, add to corpus and refit if needed
                    all_texts = self._corpus_cache + [t for t in texts if t not in self._corpus_cache]
                    
                    if len(all_texts) > len(self._corpus_cache):
                        logger.info("  🔄 Updating TF-IDF model with new texts...")
                        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                        self.svd.fit(tfidf_matrix)
                        self._corpus_cache = all_texts
                    
                    # Transform only the requested texts
                    tfidf_matrix = self.vectorizer.transform(texts)
                    embeddings = self.svd.transform(tfidf_matrix)
                
                return embeddings
        
        self.model_name = "TF-IDF + SVD"
        self.embedding_dimension = 384
        self.model_type = 'tfidf'
        logger.info(f"  ✅ Created TF-IDF fallback model")
        
        return TFIDFEmbeddingModel(384)
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        
        if not isinstance(texts, list):
            texts = [str(texts)]
        
        if not texts:
            return np.array([])
        
        logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
        
        try:
            if self.model_type == 'transformer':
                # Use sentence transformer
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            else:
                # Use TF-IDF or custom model
                embeddings = self.model.encode(texts, show_progress_bar=show_progress)
            
            logger.info(f"✅ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}D)")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.encode([text])[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        # Normalize embeddings
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def batch_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarities between a query and multiple embeddings"""
        
        # Normalize all embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Calculate batch cosine similarities
        similarities = np.dot(embeddings_norm, query_norm)
        return similarities
    
    def save_model(self, path: str) -> bool:
        """Save the current model (if applicable)"""
        
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_info = {
                'model_name': self.model_name,
                'model_type': self.model_type,
                'embedding_dimension': self.embedding_dimension
            }
            
            if self.model_type == 'tfidf':
                # Save TF-IDF components
                with open(save_path / 'tfidf_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'info': model_info
                    }, f)
                logger.info(f"💾 Saved TF-IDF model to {save_path}")
            
            # Save model info
            with open(save_path / 'model_info.json', 'w') as f:
                import json
                json.dump(model_info, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'embedding_dimension': self.embedding_dimension,
            'cache_dir': str(self.cache_dir)
        }
    
    def benchmark_speed(self, sample_texts: List[str], iterations: int = 3) -> Dict[str, float]:
        """Benchmark encoding speed"""
        
        import time
        
        logger.info(f"⚡ Benchmarking model speed with {len(sample_texts)} texts...")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            _ = self.encode(sample_texts, show_progress=False)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        texts_per_second = len(sample_texts) / avg_time
        
        benchmark_results = {
            'average_time': avg_time,
            'texts_per_second': texts_per_second,
            'total_texts': len(sample_texts),
            'iterations': iterations
        }
        
        logger.info(f"  📊 Average time: {avg_time:.3f}s")
        logger.info(f"  📊 Speed: {texts_per_second:.1f} texts/second")
        
        return benchmark_results

def setup_embedding_generator(model_name: Optional[str] = None) -> EmbeddingGenerator:
    """Main setup function for embedding generator"""
    
    logger.info("🚀 PHASE 2.2: EMBEDDING GENERATOR SETUP")
    logger.info("="*50)
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(model_name)
    
    # Display model info
    info = generator.get_model_info()
    logger.info("\n🧠 MODEL INFORMATION")
    logger.info("-" * 30)
    for key, value in info.items():
        logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    # Quick test with sample data
    test_texts = [
        "Microsoft 365 email security features",
        "Teams collaboration and communication",
        "SharePoint document management",
        "Advanced data protection capabilities"
    ]
    
    logger.info("\n🧪 TESTING EMBEDDING GENERATION")
    logger.info("-" * 40)
    
    try:
        embeddings = generator.encode(test_texts, show_progress=False)
        logger.info(f"✅ Test successful: {embeddings.shape}")
        
        # Benchmark speed
        benchmark = generator.benchmark_speed(test_texts, iterations=2)
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    logger.info("\n✅ PHASE 2.2 COMPLETE!")
    logger.info("="*50)
    
    return generator

if __name__ == "__main__":
    # Example usage and testing
    generator = setup_embedding_generator()
    
    # Display final status
    print("\n🎯 Embedding Generator Ready!")
    print("Next: Run 03_hybrid_storage.py")