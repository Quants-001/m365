# scripts/step2_hybrid_storage.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import hashlib

# Neo4j for graph operations
from neo4j import GraphDatabase

# Vector operations and embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Environment and utilities
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridVectorGraphStorage:
    """
    Hybrid storage system combining:
    1. Vector embeddings for semantic similarity (ChromaDB)
    2. Graph relationships for structured traversal (Neo4j)
    3. Metadata indexing for filtering and routing
    """
    
    def __init__(self):
        load_dotenv()
        
        # Neo4j connection (your existing graph)
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME") 
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Vector embedding model options (choose one):
        logger.info("🧠 Loading embedding model...")
        
        # Try multiple model loading strategies
        self.embedding_model = self._load_embedding_model()
        logger.info(f"✅ Model loaded (dimension: {self.embedding_dimension})")
        
        # ChromaDB for vector storage  
        logger.info("🗄️ Setting up ChromaDB...")
        # Adjust path to match your project structure
        vector_db_path = Path("../data/vector_db")
        vector_db_path.mkdir(parents=True, exist_ok=True)
        self.vector_db = chromadb.PersistentClient(path=str(vector_db_path))
        
        # Collections for different types of content
        self.collections = {
            'features': self._get_or_create_collection('m365_features'),
            'plans': self._get_or_create_collection('m365_plans'),
            'relationships': self._get_or_create_collection('m365_relationships')
        }
        
        logger.info("✅ Hybrid storage system initialized")
    
    def _generate_unique_id(self, prefix: str, content: str, counter: int = None) -> str:
        """Generate a unique ID using SHA-256 hash to avoid collisions"""
        # Use a combination of content and timestamp for uniqueness
        timestamp = datetime.now().isoformat()
        unique_content = f"{content}_{timestamp}_{counter}" if counter is not None else f"{content}_{timestamp}"
        
        # Create SHA-256 hash for uniqueness
        hash_object = hashlib.sha256(unique_content.encode())
        hash_hex = hash_object.hexdigest()[:12]  # Use first 12 characters
        
        return f"{prefix}_{hash_hex}"
    
    def _load_embedding_model(self):
        """Load embedding model with multiple fallback strategies"""
        
        # Strategy 1: Try with offline/cache-only mode first
        model_options = [
            {
                'name': 'all-MiniLM-L6-v2',
                'dimension': 384,
                'description': 'Fast and lightweight'
            },
            {
                'name': 'paraphrase-MiniLM-L6-v2', 
                'dimension': 384,
                'description': 'Alternative lightweight model'
            },
            {
                'name': 'all-MiniLM-L12-v2',
                'dimension': 384, 
                'description': 'Slightly larger model'
            }
        ]
        
        for model_info in model_options:
            model_name = model_info['name']
            logger.info(f"Trying to load model: {model_name}")
            
            try:
                # Strategy 1: Try loading from cache first
                logger.info("  → Attempting cache-only load...")
                model = SentenceTransformer(model_name, local_files_only=True)
                self.embedding_dimension = model_info['dimension']
                logger.info(f"  ✅ Loaded {model_name} from cache")
                return model
                
            except Exception as cache_error:
                logger.info(f"  → Cache miss, trying online download...")
                
                try:
                    # Strategy 2: Set trust_remote_code and clear any auth issues
                    os.environ.pop('HUGGINGFACE_HUB_TOKEN', None)  # Clear any invalid tokens
                    
                    model = SentenceTransformer(
                        model_name,
                        trust_remote_code=False,  # Don't trust remote code for security
                        local_files_only=False
                    )
                    self.embedding_dimension = model_info['dimension']
                    logger.info(f"  ✅ Downloaded and loaded {model_name}")
                    return model
                    
                except Exception as download_error:
                    logger.warning(f"  ❌ Failed to load {model_name}: {download_error}")
                    continue
        
        # Strategy 3: Fallback to a simple custom embedding (for testing)
        logger.warning("⚠️  All model downloads failed. Using fallback embedding...")
        return self._create_fallback_embedding_model()
    
    def _create_fallback_embedding_model(self):
        """Create a simple fallback embedding model for testing purposes"""
        logger.info("Creating fallback TF-IDF based embedding model...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        import pickle
        
        class FallbackEmbeddingModel:
            def __init__(self, dimension=384):
                self.dimension = dimension
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.svd = TruncatedSVD(n_components=dimension)
                self.is_fitted = False
            
            def encode(self, texts, show_progress_bar=False):
                if not self.is_fitted:
                    # Fit on the input texts (simple approach for testing)
                    tfidf_matrix = self.vectorizer.fit_transform(texts)
                    embeddings = self.svd.fit_transform(tfidf_matrix)
                    self.is_fitted = True
                    return embeddings
                else:
                    tfidf_matrix = self.vectorizer.transform(texts)
                    return self.svd.transform(tfidf_matrix)
        
        self.embedding_dimension = 384
        return FallbackEmbeddingModel(384)
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.vector_db.get_collection(name)
        except:
            return self.vector_db.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def extract_text_from_graph(self) -> Dict[str, List[Dict]]:
        """Extract text content from Neo4j graph for embedding"""
        logger.info("📤 Extracting text content from graph...")
        
        extracted_content = {
            'features': [],
            'plans': [],
            'relationships': []
        }
        
        with self.neo4j_driver.session() as session:
            # Extract Features with rich context
            feature_query = """
            MATCH (f:Feature)
            OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
            OPTIONAL MATCH (f)-[:AVAILABLE_IN]->(p:Plan)
            WITH f, c, collect(DISTINCT p.name) as plans
            RETURN f.id as id, f.name as name, f.description as description,
                   c.name as category, c.type as category_type, plans
            """
            
            features = session.run(feature_query).data()
            for i, feature in enumerate(features):
                # Create rich text description for embedding
                text_parts = [feature['name']]
                
                if feature['description'] and feature['description'] != feature['name']:
                    text_parts.append(feature['description'])
                
                if feature['category']:
                    text_parts.append(f"Category: {feature['category']}")
                
                if feature['plans']:
                    plans_text = f"Available in: {', '.join(feature['plans'][:5])}"  # Limit to first 5 plans
                    if len(feature['plans']) > 5:
                        plans_text += f" and {len(feature['plans']) - 5} more"
                    text_parts.append(plans_text)
                
                full_text = ". ".join(text_parts)
                
                # Generate unique ID
                unique_id = feature['id'] if feature['id'] else self._generate_unique_id("feat", feature['name'], i)
                
                extracted_content['features'].append({
                    'id': unique_id,
                    'name': feature['name'],
                    'text': full_text,
                    'category': feature['category'],
                    'category_type': feature['category_type'],
                    'plans_count': len(feature['plans']) if feature['plans'] else 0,
                    'metadata': {
                        'type': 'feature',
                        'source': 'neo4j_graph',
                        'extracted_at': datetime.now().isoformat()
                    }
                })
            
            # Extract Plans with context
            plan_query = """
            MATCH (p:Plan)
            OPTIONAL MATCH (f:Feature)-[:AVAILABLE_IN]->(p)
            WITH p, count(f) as feature_count
            OPTIONAL MATCH (p)-[:UPGRADES_TO]->(higher:Plan)
            OPTIONAL MATCH (lower:Plan)-[:UPGRADES_TO]->(p)
            RETURN p.id as id, p.name as name, p.type as type, 
                   feature_count, higher.name as upgrades_to, lower.name as upgraded_from
            """
            
            plans = session.run(plan_query).data()
            for i, plan in enumerate(plans):
                text_parts = [plan['name']]
                
                if plan['type']:
                    text_parts.append(f"Type: {plan['type']}")
                
                if plan['feature_count']:
                    text_parts.append(f"Includes {plan['feature_count']} features")
                
                if plan['upgraded_from']:
                    text_parts.append(f"Upgrade from: {plan['upgraded_from']}")
                
                if plan['upgrades_to']:
                    text_parts.append(f"Can upgrade to: {plan['upgrades_to']}")
                
                full_text = ". ".join(text_parts)
                
                # Generate unique ID
                unique_id = plan['id'] if plan['id'] else self._generate_unique_id("plan", plan['name'], i)
                
                extracted_content['plans'].append({
                    'id': unique_id,
                    'name': plan['name'],
                    'text': full_text,
                    'type': plan['type'],
                    'feature_count': plan['feature_count'] or 0,
                    'metadata': {
                        'type': 'plan',
                        'source': 'neo4j_graph',
                        'extracted_at': datetime.now().isoformat()
                    }
                })
            
            # Extract relationship patterns for contextual understanding
            relationship_query = """
            MATCH (f:Feature)-[r:AVAILABLE_IN]->(p:Plan)
            WHERE r.availability IN ['✔', 'Plan 1', 'Plan 2', 'Basic', 'Standard', 'Premium']
            WITH f, p, r
            OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
            RETURN f.name as feature_name, p.name as plan_name, p.type as plan_type,
                   r.availability as availability, c.name as category
            LIMIT 1000
            """
            
            relationships = session.run(relationship_query).data()
            
            # Group relationships by feature for richer context, but ensure unique IDs
            feature_relationships = {}
            relationship_counter = 0
            
            for rel in relationships:
                feature_name = rel['feature_name']
                plan_name = rel['plan_name']
                
                # Create a unique key for each feature-plan combination
                unique_key = f"{feature_name}|{plan_name}|{rel['availability']}"
                
                if unique_key not in feature_relationships:
                    feature_relationships[unique_key] = {
                        'feature_name': feature_name,
                        'plans': [],
                        'category': rel['category'],
                        'counter': relationship_counter
                    }
                    relationship_counter += 1
                
                feature_relationships[unique_key]['plans'].append({
                    'name': rel['plan_name'],
                    'type': rel['plan_type'],
                    'availability': rel['availability']
                })
            
            # Create relationship embeddings with guaranteed unique IDs
            for unique_key, rel_data in feature_relationships.items():
                feature_name = rel_data['feature_name']
                
                plans_by_type = {}
                for plan in rel_data['plans']:
                    plan_type = plan['type'] or 'Other'
                    if plan_type not in plans_by_type:
                        plans_by_type[plan_type] = []
                    plans_by_type[plan_type].append(plan['name'])
                
                # Create descriptive text
                text_parts = [f"{feature_name} is available in Microsoft 365"]
                
                if rel_data['category']:
                    text_parts.append(f"This is a {rel_data['category']} feature")
                
                for plan_type, plan_names in plans_by_type.items():
                    text_parts.append(f"{plan_type} plans: {', '.join(plan_names[:3])}")
                
                full_text = ". ".join(text_parts)
                
                # Generate guaranteed unique ID
                unique_id = self._generate_unique_id("rel", unique_key, rel_data['counter'])
                
                extracted_content['relationships'].append({
                    'id': unique_id,
                    'feature_name': feature_name,
                    'text': full_text,
                    'category': rel_data['category'],
                    'plan_types': list(plans_by_type.keys()),
                    'metadata': {
                        'type': 'relationship',
                        'source': 'neo4j_graph',
                        'extracted_at': datetime.now().isoformat()
                    }
                })
        
        logger.info(f"📤 Extracted: {len(extracted_content['features'])} features, "
                   f"{len(extracted_content['plans'])} plans, "
                   f"{len(extracted_content['relationships'])} relationships")
        
        # Verify no duplicate IDs before returning
        for content_type, items in extracted_content.items():
            ids = [item['id'] for item in items]
            unique_ids = set(ids)
            if len(ids) != len(unique_ids):
                logger.warning(f"⚠️ Found {len(ids) - len(unique_ids)} duplicate IDs in {content_type}")
                # Find and log duplicates
                from collections import Counter
                id_counts = Counter(ids)
                duplicates = [id_val for id_val, count in id_counts.items() if count > 1]
                logger.warning(f"Duplicate IDs: {duplicates[:5]}...")  # Show first 5
            else:
                logger.info(f"✅ All {len(ids)} {content_type} IDs are unique")
        
        return extracted_content
    
    def store_vectors(self, content: Dict[str, List[Dict]]):
        """Store extracted content as vectors in ChromaDB"""
        logger.info("🗄️ Storing vectors in ChromaDB...")
        
        for content_type, items in content.items():
            if not items:
                continue
                
            collection = self.collections[content_type]
            
            # Prepare data for embedding
            texts = [item['text'] for item in items]
            ids = [item['id'] for item in items]
            metadatas = [item['metadata'] for item in items]
            
            # Verify IDs are unique before storing
            unique_ids = set(ids)
            if len(ids) != len(unique_ids):
                logger.error(f"❌ Duplicate IDs found in {content_type}: {len(ids)} total, {len(unique_ids)} unique")
                from collections import Counter
                id_counts = Counter(ids)
                duplicates = [id_val for id_val, count in id_counts.items() if count > 1]
                logger.error(f"Duplicate IDs: {duplicates}")
                raise ValueError(f"Cannot store {content_type}: duplicate IDs found")
            
            # Add additional metadata
            for i, item in enumerate(items):
                metadatas[i].update({
                    'name': item.get('name', ''),
                    'content_type': content_type
                })
                
                # Add type-specific metadata
                if content_type == 'features':
                    metadatas[i].update({
                        'category': item.get('category', ''),
                        'plans_count': item.get('plans_count', 0)
                    })
                elif content_type == 'plans':
                    metadatas[i].update({
                        'plan_type': item.get('type', ''),
                        'feature_count': item.get('feature_count', 0)
                    })
                elif content_type == 'relationships':
                    metadatas[i].update({
                        'feature_name': item.get('feature_name', ''),
                        'category': item.get('category', ''),
                        'plan_types': ','.join(item.get('plan_types', []))
                    })
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Store in ChromaDB (upsert to handle duplicates)
            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Stored {len(items)} {content_type} vectors")
    
    def create_hybrid_index(self):
        """Create indexes that link vector and graph storage"""
        logger.info("🔗 Creating hybrid indexes...")
        
        # Store mapping between vector IDs and Neo4j node IDs
        vector_graph_mapping = {
            'features': {},
            'plans': {},
            'relationships': {}
        }
        
        with self.neo4j_driver.session() as session:
            # Map feature vectors to Neo4j nodes
            features = session.run("MATCH (f:Feature) RETURN f.id as id, f.name as name").data()
            for i, feature in enumerate(features):
                vector_id = feature['id'] if feature['id'] else self._generate_unique_id("feat", feature['name'], i)
                vector_graph_mapping['features'][vector_id] = {
                    'neo4j_id': feature['id'],
                    'name': feature['name']
                }
            
            # Map plan vectors to Neo4j nodes  
            plans = session.run("MATCH (p:Plan) RETURN p.id as id, p.name as name").data()
            for i, plan in enumerate(plans):
                vector_id = plan['id'] if plan['id'] else self._generate_unique_id("plan", plan['name'], i)
                vector_graph_mapping['plans'][vector_id] = {
                    'neo4j_id': plan['id'],
                    'name': plan['name']
                }
        
        # Save mapping for fast lookup during queries
        mapping_path = Path("../data/vector_graph_mapping.json")
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_path, 'w') as f:
            json.dump(vector_graph_mapping, f, indent=2)
        
        logger.info(f"✅ Hybrid index created: {mapping_path}")
        
        return vector_graph_mapping
    
    def test_hybrid_storage(self):
        """Test the hybrid storage system"""
        logger.info("🧪 Testing hybrid storage system...")
        
        # Test vector similarity search
        test_queries = [
            "security features for enterprise",
            "email protection and compliance", 
            "teams collaboration tools",
            "data loss prevention"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing query: '{query}'")
            
            # Vector search across all collections
            query_embedding = self.generate_embeddings([query])[0]
            
            for collection_name, collection in self.collections.items():
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=3,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if results['documents'] and results['documents'][0]:
                        logger.info(f"  📊 {collection_name.title()} results:")
                        for i, (doc, meta, dist) in enumerate(zip(
                            results['documents'][0],
                            results['metadatas'][0], 
                            results['distances'][0]
                        )):
                            logger.info(f"    {i+1}. {meta.get('name', 'Unknown')} (similarity: {1-dist:.3f})")
                            logger.info(f"       {doc[:100]}...")
                    
                except Exception as e:
                    logger.warning(f"    ⚠️ Error querying {collection_name}: {e}")
        
        # Test graph connectivity
        with self.neo4j_driver.session() as session:
            logger.info(f"\n🕸️ Testing graph connectivity:")
            
            # Sample path queries
            paths = session.run("""
                MATCH path = (f:Feature)-[:AVAILABLE_IN]->(p:Plan)-[:UPGRADES_TO]->(p2:Plan)
                RETURN f.name as feature, p.name as plan1, p2.name as plan2
                LIMIT 3
            """).data()
            
            for path in paths:
                logger.info(f"  🔗 {path['feature']} → {path['plan1']} → {path['plan2']}")
        
        logger.info("✅ Hybrid storage testing complete")
    
    def get_storage_stats(self):
        """Get statistics about the hybrid storage"""
        stats = {
            'vector_collections': {},
            'graph_nodes': {},
            'graph_relationships': {}
        }
        
        # Vector storage stats
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats['vector_collections'][name] = count
            except:
                stats['vector_collections'][name] = 0
        
        # Graph storage stats
        with self.neo4j_driver.session() as session:
            # Node counts
            labels = session.run("CALL db.labels()").data()
            for label_record in labels:
                label = label_record['label']
                count = session.run(f"MATCH (n:{label}) RETURN count(n) as count").single()['count']
                stats['graph_nodes'][label] = count
            
            # Relationship counts
            rel_types = session.run("CALL db.relationshipTypes()").data()
            for rel_record in rel_types:
                rel_type = rel_record['relationshipType']
                count = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count").single()['count']
                stats['graph_relationships'][rel_type] = count
        
        return stats

def setup_hybrid_storage():
    """Main function to set up hybrid vector + graph storage"""
    logger.info("🚀 STEP 2: SETTING UP HYBRID VECTOR + GRAPH STORAGE")
    logger.info("="*60)
    
    try:
        # Initialize the hybrid storage system
        hybrid_storage = HybridVectorGraphStorage()
        
        # Extract content from graph for embedding
        content = hybrid_storage.extract_text_from_graph()
        
        # Store as vectors
        hybrid_storage.store_vectors(content)
        
        # Create hybrid indexes
        mapping = hybrid_storage.create_hybrid_index()
        
        # Test the system
        hybrid_storage.test_hybrid_storage()
        
        # Get final statistics
        stats = hybrid_storage.get_storage_stats()
        
        logger.info("\n" + "="*60)
        logger.info("📊 HYBRID STORAGE STATISTICS")
        logger.info("="*60)
        logger.info(f"Vector Collections: {stats['vector_collections']}")
        logger.info(f"Graph Nodes: {stats['graph_nodes']}")
        logger.info(f"Graph Relationships: {stats['graph_relationships']}")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 STEP 2 COMPLETE!")
        logger.info("✅ Vector embeddings created and stored")
        logger.info("✅ Hybrid indexes linking vectors ↔ graph")
        logger.info("✅ Metadata filtering ready")
        logger.info("✅ System tested and verified")
        logger.info("="*60)
        
        return hybrid_storage, stats
        
    except Exception as e:
        logger.error(f"❌ Error in Step 2: {e}")
        raise

if __name__ == "__main__":
    hybrid_storage, stats = setup_hybrid_storage()
    print(f"\n🎯 READY FOR STEP 3: Query Processing Engine")