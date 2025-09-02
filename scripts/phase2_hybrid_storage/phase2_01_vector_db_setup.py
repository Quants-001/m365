# phase2_hybrid_storage/01_vector_db_setup.py
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
import logging
from typing import Dict, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBSetup:
    """
    Vector Database Setup and Management
    Handles ChromaDB initialization, collection management, and configuration
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize Vector Database Setup"""
        self.db_path = Path(db_path) if db_path else Path(r"D:\m365\data\vector_db")
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🗄️ Setting up ChromaDB at: {self.db_path}")
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collections = {}
        
        logger.info("✅ ChromaDB client initialized")
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None) -> chromadb.Collection:
        """Create a new ChromaDB collection with optimal settings"""
        
        default_metadata = {
            "hnsw:space": "cosine",  # Use cosine similarity
            "hnsw:construction_ef": 200,  # Higher for better accuracy
            "hnsw:M": 16,  # Good balance of speed/accuracy
            "created_at": datetime.now().isoformat(),
            "description": f"Microsoft 365 {name} collection"
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        try:
            # Try to get existing collection first
            collection = self.client.get_collection(name)
            logger.info(f"📁 Retrieved existing collection: {name}")
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=name,
                metadata=default_metadata
            )
            logger.info(f"✨ Created new collection: {name}")
        
        self.collections[name] = collection
        return collection
    
    def setup_m365_collections(self) -> Dict[str, chromadb.Collection]:
        """Setup standard Microsoft 365 collections"""
        
        collection_configs = {
            'm365_features': {
                'description': 'Microsoft 365 feature embeddings',
                'content_type': 'feature'
            },
            'm365_plans': {
                'description': 'Microsoft 365 plan embeddings',
                'content_type': 'plan'
            },
            'm365_relationships': {
                'description': 'Feature-plan relationship embeddings',
                'content_type': 'relationship'
            }
        }
        
        logger.info("🏗️ Setting up Microsoft 365 collections...")
        
        for collection_name, config in collection_configs.items():
            collection = self.create_collection(collection_name, config)
            logger.info(f"  ✅ {collection_name}: {collection.count()} documents")
        
        return self.collections
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """Get an existing collection"""
        try:
            if name in self.collections:
                return self.collections[name]
            
            collection = self.client.get_collection(name)
            self.collections[name] = collection
            return collection
        except Exception as e:
            logger.error(f"❌ Failed to get collection {name}: {e}")
            return None
    
    def list_collections(self) -> Dict[str, Dict]:
        """List all collections with their metadata"""
        collections_info = {}
        
        try:
            collections = self.client.list_collections()
            
            for collection in collections:
                collection_obj = self.client.get_collection(collection.name)
                collections_info[collection.name] = {
                    'count': collection_obj.count(),
                    'metadata': collection.metadata
                }
                
        except Exception as e:
            logger.error(f"❌ Error listing collections: {e}")
        
        return collections_info
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(name)
            if name in self.collections:
                del self.collections[name]
            logger.info(f"🗑️ Deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete collection {name}: {e}")
            return False
    
    def reset_all_collections(self) -> bool:
        """Reset all collections (useful for development)"""
        try:
            collections = self.client.list_collections()
            
            for collection in collections:
                self.client.delete_collection(collection.name)
                logger.info(f"🗑️ Deleted: {collection.name}")
            
            self.collections.clear()
            logger.info("🧹 All collections reset")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error resetting collections: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics"""
        stats = {
            'db_path': str(self.db_path),
            'total_collections': 0,
            'total_documents': 0,
            'collections': {}
        }
        
        try:
            collections = self.client.list_collections()
            stats['total_collections'] = len(collections)
            
            for collection in collections:
                collection_obj = self.client.get_collection(collection.name)
                count = collection_obj.count()
                
                stats['collections'][collection.name] = {
                    'document_count': count,
                    'metadata': collection.metadata
                }
                stats['total_documents'] += count
                
        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
        
        return stats
    
    def optimize_collections(self):
        """Optimize all collections for better performance"""
        logger.info("⚡ Optimizing collections...")
        
        for collection_name, collection in self.collections.items():
            try:
                # ChromaDB automatically optimizes, but we can log status
                count = collection.count()
                logger.info(f"  📊 {collection_name}: {count:,} documents")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Could not optimize {collection_name}: {e}")
        
        logger.info("✅ Collection optimization complete")

def setup_vector_database(db_path: Optional[str] = None, reset: bool = False) -> VectorDBSetup:
    """Main setup function for vector database"""
    
    logger.info("🚀 PHASE 2.1: VECTOR DATABASE SETUP")
    logger.info("="*50)
    
    # Initialize vector database
    vector_db = VectorDBSetup(db_path)
    
    if reset:
        logger.info("🧹 Resetting existing collections...")
        vector_db.reset_all_collections()
    
    # Setup Microsoft 365 specific collections
    collections = vector_db.setup_m365_collections()
    
    # Get initial stats
    stats = vector_db.get_database_stats()
    
    # Log results
    logger.info("\n📊 DATABASE STATISTICS")
    logger.info("-" * 30)
    logger.info(f"Database Path: {stats['db_path']}")
    logger.info(f"Total Collections: {stats['total_collections']}")
    logger.info(f"Total Documents: {stats['total_documents']:,}")
    
    for name, info in stats['collections'].items():
        logger.info(f"  📁 {name}: {info['document_count']:,} docs")
    
    logger.info("\n✅ PHASE 2.1 COMPLETE!")
    logger.info("="*50)
    
    return vector_db

if __name__ == "__main__":
    # Example usage and testing
    vector_db = setup_vector_database(reset=False)
    
    # Display final status
    print("\n🎯 Vector Database Ready!")
    print("Next: Run 02_embedding_generator.py")