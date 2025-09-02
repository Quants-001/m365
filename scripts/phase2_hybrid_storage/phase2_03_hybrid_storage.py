# phase2_hybrid_storage/03_hybrid_storage.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import hashlib
import logging

# Neo4j for graph operations
from neo4j import GraphDatabase

# Import our modular components
from phase2_01_vector_db_setup import VectorDBSetup
from phase2_02_embedding_generator import EmbeddingGenerator

# Environment utilities
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridStorage:
    """
    Hybrid Storage Engine
    Orchestrates vector and graph storage integration
    """
    
    def __init__(self, 
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None, 
                 neo4j_password: Optional[str] = None,
                 vector_db_path: Optional[str] = None,
                 embedding_model: Optional[str] = None):
        
        load_dotenv()
        
        # Neo4j connection
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        
        logger.info("Connecting to Neo4j...")
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize vector database
        logger.info("Initializing vector database...")
        self.vector_db = VectorDBSetup(vector_db_path)
        self.collections = self.vector_db.setup_m365_collections()
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        logger.info("HybridStorage initialized successfully")
    
    def _generate_unique_id(self, prefix: str, content: str, counter: Optional[int] = None) -> str:
        """Generate a unique ID using SHA-256 hash to avoid collisions"""
        timestamp = datetime.now().isoformat()
        unique_content = f"{content}_{timestamp}_{counter}" if counter is not None else f"{content}_{timestamp}"
        
        hash_object = hashlib.sha256(unique_content.encode())
        hash_hex = hash_object.hexdigest()[:12]
        
        return f"{prefix}_{hash_hex}"
    
    def extract_graph_content(self) -> Dict[str, List[Dict]]:
        """Extract structured content from Neo4j graph for embedding"""
        
        logger.info("Extracting content from Neo4j graph...")
        
        extracted_content = {
            'features': [],
            'plans': [],
            'relationships': []
        }
        
        with self.neo4j_driver.session() as session:
            # Extract Features with rich context
            extracted_content['features'] = self._extract_features(session)
            
            # Extract Plans with context  
            extracted_content['plans'] = self._extract_plans(session)
            
            # Extract relationship patterns
            extracted_content['relationships'] = self._extract_relationships(session)
        
        # Log extraction results
        for content_type, items in extracted_content.items():
            logger.info(f"  {content_type}: {len(items)} items")
            
            # Verify unique IDs
            ids = [item['id'] for item in items]
            unique_ids = set(ids)
            if len(ids) != len(unique_ids):
                logger.warning(f"  {len(ids) - len(unique_ids)} duplicate IDs in {content_type}")
            else:
                logger.info(f"  All {len(ids)} {content_type} IDs are unique")
        
        return extracted_content
    
    def _extract_features(self, session) -> List[Dict]:
        """Extract features from graph with rich context"""
        
        feature_query = """
        MATCH (f:Feature)
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
        OPTIONAL MATCH (f)-[:AVAILABLE_IN]->(p:Plan)
        WITH f, c, collect(DISTINCT p.name) as plans
        RETURN f.id as id, f.name as name, f.description as description,
               c.name as category, c.type as category_type, plans
        """
        
        features = session.run(feature_query).data()
        extracted_features = []
        
        for i, feature in enumerate(features):
            # Create rich text description
            text_parts = [feature['name']]
            
            if feature['description'] and feature['description'] != feature['name']:
                text_parts.append(feature['description'])
            
            if feature['category']:
                text_parts.append(f"Category: {feature['category']}")
            
            if feature['plans']:
                plans_text = f"Available in: {', '.join(feature['plans'][:5])}"
                if len(feature['plans']) > 5:
                    plans_text += f" and {len(feature['plans']) - 5} more plans"
                text_parts.append(plans_text)
            
            full_text = ". ".join(text_parts)
            
            # Generate unique ID
            unique_id = feature['id'] if feature['id'] else self._generate_unique_id("feat", feature['name'], i)
            
            extracted_features.append({
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
        
        return extracted_features
    
    def _extract_plans(self, session) -> List[Dict]:
        """Extract plans from graph with context"""
        
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
        extracted_plans = []
        
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
            
            extracted_plans.append({
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
        
        return extracted_plans
    
    def _extract_relationships(self, session) -> List[Dict]:
        """Extract relationship patterns from graph"""
        
        relationship_query = """
        MATCH (f:Feature)-[r:AVAILABLE_IN]->(p:Plan)
        WHERE r.availability IN ['✓', 'Plan 1', 'Plan 2', 'Basic', 'Standard', 'Premium']
        WITH f, p, r
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
        RETURN f.name as feature_name, p.name as plan_name, p.type as plan_type,
               r.availability as availability, c.name as category
        LIMIT 1000
        """
        
        relationships = session.run(relationship_query).data()
        extracted_relationships = []
        
        # Group relationships by feature for richer context
        feature_relationships = {}
        relationship_counter = 0
        
        for rel in relationships:
            feature_name = rel['feature_name']
            plan_name = rel['plan_name']
            
            # Create unique key for feature-plan combination
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
        
        # Create relationship embeddings
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
            
            # Generate unique ID
            unique_id = self._generate_unique_id("rel", unique_key, rel_data['counter'])
            
            extracted_relationships.append({
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
        
        return extracted_relationships
    
    def store_embeddings(self, content: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Generate and store embeddings for extracted content"""
        
        logger.info("Generating and storing embeddings...")
        
        storage_stats = {}
        
        for content_type, items in content.items():
            if not items:
                storage_stats[content_type] = 0
                continue
            
            logger.info(f"  Processing {content_type}...")
            
            collection = self.collections[f'm365_{content_type}']
            
            # Prepare data for embedding
            texts = [item['text'] for item in items]
            ids = [item['id'] for item in items]
            
            # Verify unique IDs before storage
            unique_ids = set(ids)
            if len(ids) != len(unique_ids):
                raise ValueError(f"Duplicate IDs found in {content_type}")
            
            # Generate embeddings
            embeddings = self.embedding_generator.encode(texts, show_progress=True)
            
            # Prepare metadata for ChromaDB - FIXED to handle None values
            metadatas = []
            for item in items:
                metadata = item['metadata'].copy()
                
                # Helper function to clean metadata values
                def clean_value(value):
                    """Clean metadata values to ensure ChromaDB compatibility"""
                    if value is None:
                        return ""  # Convert None to empty string
                    elif isinstance(value, (str, int, float, bool)):
                        return value
                    elif isinstance(value, list):
                        # Convert list to comma-separated string
                        return ",".join(str(v) for v in value if v is not None)
                    else:
                        return str(value)  # Convert other types to string
                
                # Clean base metadata
                metadata['name'] = clean_value(item.get('name', ''))
                metadata['content_type'] = clean_value(content_type)
                metadata['type'] = clean_value(metadata.get('type', ''))
                metadata['source'] = clean_value(metadata.get('source', ''))
                metadata['extracted_at'] = clean_value(metadata.get('extracted_at', ''))
                
                # Add type-specific metadata with None handling
                if content_type == 'features':
                    metadata['category'] = clean_value(item.get('category', ''))
                    metadata['category_type'] = clean_value(item.get('category_type', ''))
                    metadata['plans_count'] = item.get('plans_count', 0) or 0  # Ensure it's an integer
                    
                elif content_type == 'plans':
                    metadata['plan_type'] = clean_value(item.get('type', ''))
                    metadata['feature_count'] = item.get('feature_count', 0) or 0  # Ensure it's an integer
                    
                elif content_type == 'relationships':
                    metadata['feature_name'] = clean_value(item.get('feature_name', ''))
                    metadata['category'] = clean_value(item.get('category', ''))
                    metadata['plan_types'] = clean_value(','.join(item.get('plan_types', [])))
                
                # Final cleanup - remove any remaining None values
                cleaned_metadata = {}
                for key, value in metadata.items():
                    cleaned_value = clean_value(value)
                    cleaned_metadata[key] = cleaned_value
                
                metadatas.append(cleaned_metadata)
            
            # Store in ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
            
            storage_stats[content_type] = len(items)
            logger.info(f"    Stored {len(items)} {content_type} embeddings")
        
        return storage_stats
    
    def query_hybrid(self, 
                     query: str,
                     content_types: Optional[List[str]] = None,
                     n_results: int = 5,
                     similarity_threshold: float = 0.0) -> Dict[str, List[Dict]]:
        """Query both vector and graph storage"""
        
        logger.info(f"Hybrid query: '{query}'")
        
        # Default to all content types
        if content_types is None:
            content_types = ['features', 'plans', 'relationships']
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode([query])[0]
        
        results = {}
        
        for content_type in content_types:
            if content_type not in self.collections:
                continue
            
            collection_name = f'm365_{content_type}'
            collection = self.collections[collection_name]
            
            try:
                # Vector similarity search
                search_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Process results
                processed_results = []
                if search_results['documents'] and search_results['documents'][0]:
                    for i, (doc, meta, dist) in enumerate(zip(
                        search_results['documents'][0],
                        search_results['metadatas'][0],
                        search_results['distances'][0]
                    )):
                        similarity = 1 - dist  # Convert distance to similarity
                        if similarity >= similarity_threshold:
                            processed_results.append({
                                'text': doc,
                                'metadata': meta,
                                'similarity': similarity,
                                'rank': i + 1
                            })
                
                results[content_type] = processed_results
                logger.info(f"  {content_type}: {len(processed_results)} results")
                
            except Exception as e:
                logger.warning(f"  Error querying {content_type}: {e}")
                results[content_type] = []
        
        return results
    
    def get_graph_context(self, item_name: str, item_type: str = 'Feature') -> Dict[str, Any]:
        """Get rich context from graph for a specific item"""
        
        with self.neo4j_driver.session() as session:
            if item_type == 'Feature':
                query = """
                MATCH (f:Feature {name: $name})
                OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (f)-[:AVAILABLE_IN]->(p:Plan)
                OPTIONAL MATCH (f)-[:REQUIRES]->(req:Feature)
                OPTIONAL MATCH (dep:Feature)-[:REQUIRES]->(f)
                RETURN f, c, collect(DISTINCT p) as plans, 
                       collect(DISTINCT req) as requirements,
                       collect(DISTINCT dep) as dependents
                """
            elif item_type == 'Plan':
                query = """
                MATCH (p:Plan {name: $name})
                OPTIONAL MATCH (f:Feature)-[:AVAILABLE_IN]->(p)
                OPTIONAL MATCH (p)-[:UPGRADES_TO]->(higher:Plan)
                OPTIONAL MATCH (lower:Plan)-[:UPGRADES_TO]->(p)
                RETURN p, collect(DISTINCT f) as features, higher, lower
                """
            else:
                return {}
            
            result = session.run(query, name=item_name).single()
            
            if not result:
                return {}
            
            # Convert Neo4j result to dictionary
            context = {}
            for key, value in result.items():
                if hasattr(value, 'items'):  # Node object
                    context[key] = dict(value.items())
                elif isinstance(value, list):
                    context[key] = [dict(item.items()) if hasattr(item, 'items') else item for item in value]
                else:
                    context[key] = value
            
            return context
    
    def test_hybrid_system(self):
        """Comprehensive test of the hybrid storage system"""
        
        logger.info("Testing hybrid storage system...")
        
        test_queries = [
            "security features for enterprise",
            "email protection and compliance",
            "teams collaboration tools", 
            "data loss prevention",
            "SharePoint document management"
        ]
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            
            # Test hybrid query
            results = self.query_hybrid(query, n_results=3)
            
            for content_type, items in results.items():
                if items:
                    logger.info(f"  {content_type.title()}:")
                    for item in items[:2]:  # Show top 2 results
                        name = item['metadata'].get('name', 'Unknown')
                        similarity = item['similarity']
                        logger.info(f"    • {name} (similarity: {similarity:.3f})")
                else:
                    logger.info(f"  {content_type.title()}: No results")
        
        # Test graph context retrieval
        logger.info(f"\nTesting graph context...")
        context = self.get_graph_context("Microsoft Teams", "Feature")
        if context:
            logger.info(f"  Retrieved context for Microsoft Teams")
        else:
            logger.info(f"  No context found for Microsoft Teams")
        
        logger.info("Hybrid system testing complete")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        
        stats = {
            'vector_storage': {},
            'graph_storage': {},
            'hybrid_metrics': {}
        }
        
        # Vector storage stats
        vector_stats = self.vector_db.get_database_stats()
        stats['vector_storage'] = vector_stats
        
        # Graph storage stats
        with self.neo4j_driver.session() as session:
            # Node counts
            node_stats = {}
            labels = session.run("CALL db.labels()").data()
            for label_record in labels:
                label = label_record['label']
                count = session.run(f"MATCH (n:{label}) RETURN count(n) as count").single()['count']
                node_stats[label] = count
            
            # Relationship counts
            rel_stats = {}
            rel_types = session.run("CALL db.relationshipTypes()").data()
            for rel_record in rel_types:
                rel_type = rel_record['relationshipType']
                count = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count").single()['count']
                rel_stats[rel_type] = count
            
            stats['graph_storage'] = {
                'nodes': node_stats,
                'relationships': rel_stats
            }
        
        # Hybrid metrics
        stats['hybrid_metrics'] = {
            'embedding_model': self.embedding_generator.get_model_info(),
            'total_embeddings': sum(vector_stats.get('collections', {}).get(name, {}).get('document_count', 0) 
                                   for name in ['m365_features', 'm365_plans', 'm365_relationships']),
            'total_graph_nodes': sum(node_stats.values()),
            'total_graph_relationships': sum(rel_stats.values())
        }
        
        return stats
    
    def close(self):
        """Close all connections"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
        logger.info("Connections closed")

def setup_hybrid_storage(**kwargs) -> HybridStorage:
    """Main setup function for hybrid storage system"""
    
    logger.info("PHASE 2.3: HYBRID STORAGE SETUP")
    logger.info("="*50)
    
    try:
        # Initialize hybrid storage
        hybrid_storage = HybridStorage(**kwargs)
        
        # Extract content from graph
        content = hybrid_storage.extract_graph_content()
        
        # Generate and store embeddings
        storage_stats = hybrid_storage.store_embeddings(content)
        
        # Test the system
        hybrid_storage.test_hybrid_system()
        
        # Get comprehensive statistics
        stats = hybrid_storage.get_storage_statistics()
        
        # Log results
        logger.info("\nHYBRID STORAGE STATISTICS")
        logger.info("-" * 40)
        logger.info(f"Vector Collections:")
        for name, info in stats['vector_storage'].get('collections', {}).items():
            logger.info(f"  {name}: {info['document_count']:,} documents")
        
        logger.info(f"\nGraph Storage:")
        for label, count in stats['graph_storage']['nodes'].items():
            logger.info(f"  {label}: {count:,} nodes")
        
        logger.info(f"\nHybrid Metrics:")
        logger.info(f"  Model: {stats['hybrid_metrics']['embedding_model']['model_name']}")
        logger.info(f"  Total Embeddings: {stats['hybrid_metrics']['total_embeddings']:,}")
        logger.info(f"  Graph Nodes: {stats['hybrid_metrics']['total_graph_nodes']:,}")
        logger.info(f"  Graph Relations: {stats['hybrid_metrics']['total_graph_relationships']:,}")
        
        logger.info("\nPHASE 2.3 COMPLETE!")
        logger.info("="*50)
        
        return hybrid_storage
        
    except Exception as e:
        logger.error(f"Error in Phase 2.3: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    hybrid_storage = setup_hybrid_storage()
    
    # Display final status
    print("\nHybrid Storage Ready!")
    print("Next: Run 04_vector_graph_bridge.py")