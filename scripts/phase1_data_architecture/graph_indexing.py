# scripts/phase1_data_architecture/graph_indexing.py
import os
import gc

# Memory optimization for Neo4j driver
os.environ['NEO4J_MAX_CONNECTION_LIFETIME'] = '3600'
os.environ['NEO4J_MAX_CONNECTION_POOL_SIZE'] = '50'
os.environ['NEO4J_CONNECTION_ACQUISITION_TIMEOUT'] = '60'

# Force garbage collection
gc.collect()

from neo4j import GraphDatabase
from dotenv import load_dotenv
from typing import List, Dict, Any
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphIndexManager:
    """
    Advanced indexing and search utilities for Neo4j knowledge graph.
    Provides methods to create, verify, and test indexes for a RAG system.
    """
    
    def __init__(self):
        load_dotenv()
        
        # Neo4j connection with memory optimizations
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Missing Neo4j credentials in .env file")
        
        # Create driver with proper configuration based on URI scheme
        try:
            # Determine if we need encryption settings based on URI
            uri_lower = self.neo4j_uri.lower()
            needs_encryption_config = uri_lower.startswith(('bolt://', 'neo4j://'))
            
            if needs_encryption_config:
                # For bolt:// and neo4j:// schemes, we can set encryption
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60,
                    encrypted=False  # Only for local development
                )
            else:
                # For other schemes (bolt+s://, neo4j+s://, etc.), use minimal config
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=60
                )
            logger.info("Neo4j driver created successfully")
        except Exception as e:
            logger.error(f"Failed to create Neo4j driver: {e}")
            logger.info("Trying with minimal configuration...")
            # Fallback to absolute minimal configuration
            try:
                self.driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                logger.info("Neo4j driver created with minimal config")
            except Exception as e2:
                logger.error(f"Minimal configuration also failed: {e2}")
                raise
        
        # Index configuration using modern, robust syntax
        self.indexes = {
            'node_indexes': [
                {'label': 'Feature', 'property': 'name'},
                {'label': 'Feature', 'property': 'id'},
                {'label': 'Plan', 'property': 'name'},
                {'label': 'Plan', 'property': 'id'},
                {'label': 'Plan', 'property': 'type'},
                {'label': 'Category', 'property': 'name'},
                {'label': 'Category', 'property': 'type'},
                {'label': 'Concept', 'property': 'name'},
                {'label': 'Document', 'property': 'title'}
            ],
            'fulltext_indexes': [
                {
                    'name': 'entitySearch',
                    'labels': ['Feature', 'Plan', 'Category', 'Concept'],
                    'properties': ['name', 'description']
                },
                {
                    'name': 'documentSearch',
                    'labels': ['Document'],
                    'properties': ['title', 'description', 'content']
                },
                {
                    'name': 'featureSearch',
                    'labels': ['Feature'],
                    'properties': ['name', 'description']
                }
            ]
        }
        logger.info("Graph Index Manager initialized")

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")
            # Force garbage collection
            gc.collect()

    def test_connection(self):
        """Test the Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                logger.info(f"Connection test successful: {test_value}")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def create_all_indexes(self):
        """Create all necessary indexes for optimal performance"""
        logger.info("Creating all graph indexes...")
        
        if not self.test_connection():
            logger.error("Cannot create indexes - connection failed")
            return False
        
        try:
            with self.driver.session() as session:
                # Create node indexes first
                for index_config in self.indexes['node_indexes']:
                    self._create_node_index(session, index_config['label'], index_config['property'])
                    time.sleep(0.1)  # Small delay to prevent memory pressure
                
                # Create fulltext indexes
                for index_config in self.indexes['fulltext_indexes']:
                    self._create_fulltext_index(
                        session,
                        index_config['name'],
                        index_config['labels'],
                        index_config['properties']
                    )
                    time.sleep(0.1)  # Small delay
                    
            logger.info("All indexes have been created or verified successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            return False

    def _create_node_index(self, session, label: str, property: str):
        """Create a single node property index."""
        index_name = f"idx_{label.lower()}_{property.lower()}"
        query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property})"
        try:
            session.run(query)
            logger.info(f"✅ Created or verified index: {index_name}")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning for {index_name}: {e}")

    def _create_fulltext_index(self, session, name: str, labels: List[str], properties: List[str]):
        """Create a full-text search index with fallback for older Neo4j versions."""
        try:
            # Try modern syntax first
            try:
                check_query = "SHOW FULLTEXT INDEXES YIELD name WHERE name = $name"
                result = list(session.run(check_query, name=name))
                if result:
                    logger.info(f"✅ Full-text index '{name}' already exists.")
                    return
            except:
                # Fallback for older Neo4j versions
                logger.info(f"Using fallback method for fulltext index check: {name}")
        
            # Try creating with modern syntax
            try:
                labels_str = "|".join(labels)
                properties_list = [f"n.{prop}" for prop in properties]
                properties_str = ', '.join(properties_list)
                query = f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{labels_str}) ON EACH [{properties_str}]"
                session.run(query)
                logger.info(f"✅ Created full-text index: '{name}'")
                return
            except Exception as modern_error:
                logger.warning(f"Modern fulltext syntax failed: {modern_error}")
            
            # Fallback to APOC/legacy method
            try:
                labels_str = ', '.join([f"'{label}'" for label in labels])
                properties_str = ', '.join([f"'{prop}'" for prop in properties])
                query = f"CALL db.index.fulltext.createNodeIndex('{name}', [{labels_str}], [{properties_str}])"
                session.run(query)
                logger.info(f"✅ Created full-text index (legacy): '{name}'")
            except Exception as legacy_error:
                logger.error(f"❌ Failed to create full-text index '{name}': {legacy_error}")
                logger.warning("Consider upgrading Neo4j or installing APOC plugin")
                
        except Exception as e:
            logger.error(f"❌ Unexpected error creating full-text index '{name}': {e}")

    def search_features(self, search_term: str, limit: int = 5) -> List[Dict]:
        """Search for features using the full-text index with a fallback."""
        with self.driver.session() as session:
            try:
                search_query = """
                CALL db.index.fulltext.queryNodes('featureSearch', $search_term)
                YIELD node, score
                OPTIONAL MATCH (node)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (node)-[:AVAILABLE_IN]->(p:Plan)
                WITH node, score, c, collect(DISTINCT p.name) as plans
                RETURN node.name as name, node.description as description,
                       c.name as category, plans, score
                ORDER BY score DESC
                LIMIT $limit
                """
                result = session.run(search_query, search_term=search_term, limit=limit)
                return [record.data() for record in result]
            except Exception as e:
                logger.warning(f"Full-text search failed, using fallback CONTAINS search: {e}")
                return self._fallback_feature_search(session, search_term, limit)

    def _fallback_feature_search(self, session, search_term: str, limit: int) -> List[Dict]:
        """Fallback search method using CONTAINS for feature names and descriptions."""
        search_query = """
        MATCH (f:Feature)
        WHERE toLower(f.name) CONTAINS toLower($search_term)
           OR toLower(coalesce(f.description, '')) CONTAINS toLower($search_term)
        OPTIONAL MATCH (f)-[:BELONGS_TO]->(c:Category)
        OPTIONAL MATCH (f)-[:AVAILABLE_IN]->(p:Plan)
        WITH f, c, collect(DISTINCT p.name) as plans
        RETURN f.name as name, coalesce(f.description, '') as description,
               c.name as category, plans
        LIMIT $limit
        """
        result = session.run(search_query, search_term=search_term, limit=limit)
        return [record.data() for record in result]

    def get_index_status(self) -> Dict:
        """Get the status of all indexes"""
        with self.driver.session() as session:
            try:
                # Try modern syntax
                result = session.run("SHOW INDEXES")
                indexes = []
                for record in result:
                    indexes.append({
                        'name': record.get('name', 'Unknown'),
                        'state': record.get('state', 'Unknown'),
                        'type': record.get('type', 'Unknown')
                    })
                return {'indexes': indexes, 'method': 'modern'}
            except:
                # Fallback for older versions
                try:
                    result = session.run("CALL db.indexes()")
                    indexes = []
                    for record in result:
                        indexes.append({
                            'name': record.get('name', 'Unknown'),
                            'state': record.get('state', 'Unknown'),
                            'type': record.get('type', 'Unknown')
                        })
                    return {'indexes': indexes, 'method': 'legacy'}
                except Exception as e:
                    logger.warning(f"Could not get index status: {e}")
                    return {'indexes': [], 'error': str(e)}

def main():
    """Main function to instantiate the manager and run its methods."""
    manager = None
    try:
        # Force garbage collection before starting
        gc.collect()
        
        logger.info("="*60)
        logger.info("GRAPH INDEXING SETUP")
        logger.info("="*60)
        
        manager = GraphIndexManager()
        
        # Step 1: Test connection
        logger.info("Testing Neo4j connection...")
        if not manager.test_connection():
            logger.error("Cannot proceed without Neo4j connection")
            return
        
        # Step 2: Create all defined indexes
        logger.info("Creating indexes...")
        success = manager.create_all_indexes()
        
        if not success:
            logger.error("Index creation failed")
            return
        
        # Step 3: Get index status
        logger.info("Checking index status...")
        status = manager.get_index_status()
        logger.info(f"Found {len(status['indexes'])} indexes")
        
        # Step 4: Perform a test search to verify functionality
        logger.info("\n" + "="*60)
        logger.info("🧪 Performing a test search for 'security' features...")
        logger.info("="*60)
        
        search_results = manager.search_features("security")
        
        if not search_results:
            logger.info("No results found for 'security'. The index might be empty or data not loaded.")
        else:
            logger.info(f"Found {len(search_results)} results:")
            for i, result in enumerate(search_results):
                logger.info(f"  {i+1}. Feature: {result.get('name')}")
                logger.info(f"     Category: {result.get('category', 'N/A')}")
                score = result.get('score')
                if score is not None:
                    logger.info(f"     Score: {score:.3f}")

        logger.info("\n" + "="*60)
        logger.info("✅ GRAPH INDEXING COMPLETE!")
        logger.info("="*60)

    except MemoryError as me:
        logger.error(f"Memory error occurred: {me}")
        logger.error("Try running with more memory or downgrading neo4j driver:")
        logger.error("pip uninstall neo4j && pip install neo4j==4.4.12")
    except Exception as e:
        logger.error(f"An error occurred during the indexing process: {e}")
    finally:
        if manager:
            manager.close()
        # Final garbage collection
        gc.collect()

if __name__ == "__main__":
    main()