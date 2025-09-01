# scripts/neo4j_connection_test.py
"""
Quick diagnostic script to test Neo4j connection and identify configuration issues.
Run this before running the main graph indexing script.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment():
    """Test environment setup"""
    logger.info("="*50)
    logger.info("NEO4J ENVIRONMENT TEST")
    logger.info("="*50)
    
    # Load environment
    env_path = Path(".env")
    if env_path.exists():
        logger.info(f"✅ Found .env file: {env_path.resolve()}")
        load_dotenv()
    else:
        logger.error(f"❌ .env file not found at: {env_path.resolve()}")
        return False
    
    # Check environment variables
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    logger.info("\nEnvironment Variables:")
    logger.info(f"  NEO4J_URI: {neo4j_uri}")
    logger.info(f"  NEO4J_USERNAME: {neo4j_user}")
    logger.info(f"  NEO4J_PASSWORD: {'*' * len(neo4j_password) if neo4j_password else None}")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        logger.error("❌ Missing Neo4j credentials")
        return False
    
    logger.info("✅ Environment variables found")
    return True, neo4j_uri, neo4j_user, neo4j_password

def test_neo4j_import():
    """Test Neo4j Python driver import"""
    logger.info("\nTesting Neo4j import...")
    try:
        from neo4j import GraphDatabase
        logger.info("✅ Neo4j driver imported successfully")
        return True, GraphDatabase
    except MemoryError as e:
        logger.error(f"❌ Memory error importing Neo4j: {e}")
        logger.error("Solution: pip uninstall neo4j && pip install neo4j==4.4.12")
        return False, None
    except Exception as e:
        logger.error(f"❌ Error importing Neo4j: {e}")
        return False, None

def test_connection_configs(uri, user, password, GraphDatabase):
    """Test different connection configurations"""
    logger.info(f"\nTesting connection configurations for URI: {uri}")
    
    configs_to_test = [
        {
            "name": "Minimal Config",
            "config": {}
        },
        {
            "name": "Basic Config", 
            "config": {
                "max_connection_lifetime": 3600,
                "max_connection_pool_size": 10,
                "connection_acquisition_timeout": 60
            }
        }
    ]
    
    # Add encryption config only for certain URI schemes
    uri_lower = uri.lower()
    if uri_lower.startswith(('bolt://', 'neo4j://')):
        configs_to_test.append({
            "name": "With Encryption Disabled",
            "config": {
                "encrypted": False,
                "max_connection_lifetime": 3600
            }
        })
    
    successful_config = None
    
    for config_info in configs_to_test:
        logger.info(f"\n  Testing: {config_info['name']}")
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password), **config_info['config'])
            
            # Test actual connection
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()['test']
                logger.info(f"    ✅ Connection successful: {test_value}")
                successful_config = config_info
                driver.close()
                break
                
        except Exception as e:
            logger.warning(f"    ❌ Failed: {e}")
            continue
    
    return successful_config

def test_basic_queries(uri, user, password, GraphDatabase, config):
    """Test basic database queries"""
    logger.info(f"\nTesting basic database operations...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password), **config['config'])
        
        with driver.session() as session:
            # Test 1: Basic return
            result = session.run("RETURN 1 as number, 'Hello' as text")
            record = result.single()
            logger.info(f"  ✅ Basic query: {record['number']}, {record['text']}")
            
            # Test 2: Check existing data
            result = session.run("MATCH (n) RETURN count(n) as node_count LIMIT 1")
            count = result.single()['node_count']
            logger.info(f"  📊 Total nodes in database: {count}")
            
            # Test 3: Check labels
            result = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
            labels = result.single()['labels']
            logger.info(f"  🏷️  Available labels: {labels}")
            
            # Test 4: Sample features
            result = session.run("MATCH (f:Feature) RETURN f.name LIMIT 3")
            features = [record['f.name'] for record in result]
            if features:
                logger.info(f"  📋 Sample features: {features}")
            else:
                logger.info("  ⚠️  No features found - run data loading first")
                
        driver.close()
        logger.info("✅ Database operations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations failed: {e}")
        return False

def main():
    """Main diagnostic function"""
    try:
        # Test 1: Environment
        env_result = test_environment()
        if not env_result:
            return False
        
        success, uri, user, password = env_result
        if not success:
            return False
        
        # Test 2: Import
        import_success, GraphDatabase = test_neo4j_import()
        if not import_success:
            return False
        
        # Test 3: Connection configs
        successful_config = test_connection_configs(uri, user, password, GraphDatabase)
        if not successful_config:
            logger.error("❌ No working connection configuration found")
            return False
        
        logger.info(f"\n✅ Working configuration: {successful_config['name']}")
        
        # Test 4: Database operations
        db_success = test_basic_queries(uri, user, password, GraphDatabase, successful_config)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*50)
        
        if db_success:
            logger.info("✅ All tests passed!")
            logger.info(f"✅ Recommended config: {successful_config['name']}")
            logger.info(f"✅ Config details: {successful_config['config']}")
            logger.info("\nYou can now run the graph indexing script.")
        else:
            logger.error("❌ Some tests failed - check the logs above")
            
        return db_success
        
    except Exception as e:
        logger.error(f"❌ Diagnostic failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\nTroubleshooting tips:")
        print("1. Make sure Neo4j is running")
        print("2. Check your .env file has correct credentials")
        print("3. Try: pip uninstall neo4j && pip install neo4j==4.4.12")
        print("4. Make sure your Neo4j URI is correct (usually bolt://localhost:7687)")