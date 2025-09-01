# scripts/complete_step1_knowledge_graph.py
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import json

load_dotenv()

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PWD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(URI, auth=(USER, PWD))

class KnowledgeGraphBuilder:
    def __init__(self, driver):
        self.driver = driver
    
    def create_domain_entities(self):
        """Create domain-specific entities beyond just Feature-Plan"""
        with self.driver.session() as session:
            print("🏗️ Creating Domain Entities...")
            
            # Create Categories/Products (grouping features)
            categories = [
                {"id": "security", "name": "Security & Compliance", "type": "Security"},
                {"id": "productivity", "name": "Productivity Apps", "type": "Productivity"},
                {"id": "communication", "name": "Communication & Collaboration", "type": "Communication"},
                {"id": "analytics", "name": "Analytics & Insights", "type": "Analytics"},
                {"id": "management", "name": "Device & Identity Management", "type": "Management"},
                {"id": "storage", "name": "Storage & File Services", "type": "Storage"}
            ]
            
            for cat in categories:
                session.run("""
                    MERGE (c:Category {id: $id})
                    SET c.name = $name, c.type = $type
                """, **cat)
            
            print(f"✅ Created {len(categories)} domain categories")
            
            # Create concept entities (abstract concepts)
            concepts = [
                {"id": "ediscovery", "name": "eDiscovery", "description": "Electronic discovery for legal compliance"},
                {"id": "dlp", "name": "Data Loss Prevention", "description": "Preventing unauthorized data sharing"},
                {"id": "conditional_access", "name": "Conditional Access", "description": "Context-based access control"},
                {"id": "mfa", "name": "Multi-Factor Authentication", "description": "Additional authentication layers"},
                {"id": "intune", "name": "Microsoft Intune", "description": "Mobile device management"}
            ]
            
            for concept in concepts:
                session.run("""
                    MERGE (c:Concept {id: $id})
                    SET c.name = $name, c.description = $description
                """, **concept)
            
            print(f"✅ Created {len(concepts)} concept entities")
    
    def create_semantic_relationships(self):
        """Create semantic relationships between entities"""
        with self.driver.session() as session:
            print("🔗 Creating Semantic Relationships...")
            
            # Feature -> Category relationships (based on feature names)
            feature_categories = [
                ("audit", "security"),
                ("compliance", "security"),
                ("defender", "security"),
                ("dlp", "security"),
                ("conditional access", "security"),
                ("teams", "communication"),
                ("sharepoint", "productivity"),
                ("onedrive", "storage"),
                ("exchange", "communication"),
                ("viva", "analytics"),
                ("copilot", "productivity"),
                ("planner", "productivity"),
                ("bookings", "productivity")
            ]
            
            relationship_count = 0
            for keyword, category_id in feature_categories:
                result = session.run("""
                    MATCH (f:Feature)
                    MATCH (c:Category {id: $category_id})
                    WHERE toLower(f.name) CONTAINS $keyword
                    MERGE (f)-[:BELONGS_TO]->(c)
                    RETURN count(*) as count
                """, keyword=keyword, category_id=category_id)
                
                count = result.single()['count']
                relationship_count += count
                print(f"  📎 {keyword} features → {category_id}: {count} relationships")
            
            # Plan -> Plan relationships (hierarchy)
            session.run("""
                MATCH (basic:Plan), (premium:Plan)
                WHERE basic.name CONTAINS 'Basic' AND premium.name CONTAINS 'Premium'
                AND basic.type = premium.type
                MERGE (basic)-[:UPGRADES_TO]->(premium)
            """)
            
            # E-series hierarchy
            session.run("""
                MATCH (e1:Plan {name: 'E1'}), (e3:Plan {name: 'E3'}), (e5:Plan {name: 'E5'})
                MERGE (e1)-[:UPGRADES_TO]->(e3)
                MERGE (e3)-[:UPGRADES_TO]->(e5)
            """)
            
            print(f"✅ Created {relationship_count} semantic relationships")
    
    def create_document_entities(self):
        """Create document entities from metadata"""
        with self.driver.session() as session:
            print("📄 Creating Document Entities...")
            
            # Load metadata to create document entities
            data_path = Path("../kb/master_data_fixed.csv")
            if not data_path.exists():
                print("⚠️ Fixed data not found, skipping document creation")
                return
                
            df = pd.read_csv(data_path, index_col=0)
            
            # Create document representing the M365 feature matrix
            doc_info = {
                "id": "m365_feature_matrix",
                "title": "Microsoft 365 Feature Comparison Matrix",
                "type": "Feature Matrix",
                "source": "master_data.csv",
                "features_count": len(df.index),
                "plans_count": len(df.columns),
                "description": f"Comprehensive matrix of {len(df.index)} Microsoft 365 features across {len(df.columns)} different plans"
            }
            
            session.run("""
                MERGE (d:Document {id: $id})
                SET d.title = $title,
                    d.type = $type,
                    d.source = $source,
                    d.features_count = $features_count,
                    d.plans_count = $plans_count,
                    d.description = $description
            """, **doc_info)
            
            # Link features to document
            session.run("""
                MATCH (f:Feature)
                MATCH (d:Document {id: 'm365_feature_matrix'})
                MERGE (f)-[:DOCUMENTED_IN]->(d)
            """)
            
            print("✅ Created document entities and relationships")
    
    def create_indexes_for_retrieval(self):
        """Create indexes for fast retrieval in RAG system"""
        with self.driver.session() as session:
            print("🔍 Creating Retrieval Indexes...")
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (f:Feature) ON (f.name)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Plan) ON (p.name)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Plan) ON (p.type)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Category) ON (c.type)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.type)",
                # Full-text search indexes for RAG
                "CALL db.index.fulltext.createNodeIndex('entitySearch', ['Feature', 'Plan', 'Category', 'Concept'], ['name', 'description']) YIELD name RETURN 'Created: ' + name",
                "CALL db.index.fulltext.createNodeIndex('documentSearch', ['Document'], ['title', 'description', 'source']) YIELD name RETURN 'Created: ' + name"
            ]
            
            for index in indexes:
                try:
                    result = session.run(index)
                    if result.peek():
                        print(f"  ✅ {result.single().values()[0]}")
                    else:
                        print(f"  ✅ Index created successfully")
                except Exception as e:
                    print(f"  ⚠️ Index warning: {e}")
    
    def implement_entity_linking(self):
        """Implement entity linking strategies"""
        with self.driver.session() as session:
            print("🔗 Implementing Entity Linking...")
            
            # Link similar features (name similarity)
            session.run("""
                MATCH (f1:Feature), (f2:Feature)
                WHERE f1 <> f2 
                AND (
                    toLower(f1.name) CONTAINS toLower(f2.name) OR
                    toLower(f2.name) CONTAINS toLower(f1.name)
                )
                AND size(f1.name) > 5 AND size(f2.name) > 5
                MERGE (f1)-[:SIMILAR_TO]-(f2)
            """)
            
            # Link concepts to features
            concept_links = [
                ("eDiscovery", "ediscovery"),
                ("DLP", "dlp"),
                ("Conditional Access", "conditional_access"),
                ("Multi-Factor", "mfa"),
                ("Intune", "intune")
            ]
            
            for feature_keyword, concept_id in concept_links:
                session.run("""
                    MATCH (f:Feature)
                    MATCH (c:Concept {id: $concept_id})
                    WHERE toLower(f.name) CONTAINS toLower($keyword)
                    MERGE (f)-[:IMPLEMENTS]->(c)
                """, keyword=feature_keyword, concept_id=concept_id)
            
            print("✅ Entity linking completed")
    
    def create_knowledge_graph_summary(self):
        """Generate summary of the knowledge graph"""
        with self.driver.session() as session:
            print("\n" + "="*60)
            print("📊 KNOWLEDGE GRAPH SUMMARY")
            print("="*60)
            
            # Count all node types
            node_counts = {}
            labels = session.run("CALL db.labels()").data()
            for label_record in labels:
                label = label_record['label']
                count = session.run(f"MATCH (n:{label}) RETURN count(n) as count").single()['count']
                node_counts[label] = count
                print(f"  {label} nodes: {count}")
            
            # Count relationships
            print(f"\nRelationships:")
            rel_types = session.run("CALL db.relationshipTypes()").data()
            total_rels = 0
            for rel_record in rel_types:
                rel_type = rel_record['relationshipType']
                count = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count").single()['count']
                total_rels += count
                print(f"  {rel_type}: {count}")
            
            print(f"\nTotal: {sum(node_counts.values())} nodes, {total_rels} relationships")
            
            # Sample complex queries for RAG
            print(f"\n🔍 Sample Knowledge Queries:")
            
            # Features by category
            security_features = session.run("""
                MATCH (f:Feature)-[:BELONGS_TO]->(c:Category {type: 'Security'})
                RETURN count(f) as count
            """).single()
            if security_features:
                print(f"  Security features: {security_features['count']}")
            
            # Plan upgrade paths
            upgrades = session.run("""
                MATCH (p1:Plan)-[:UPGRADES_TO]->(p2:Plan)
                RETURN p1.name + ' → ' + p2.name as path
                LIMIT 3
            """).data()
            for upgrade in upgrades:
                print(f"  Upgrade path: {upgrade['path']}")
            
            return node_counts, total_rels

def complete_step1():
    """Complete Step 1: Knowledge Graph Setup"""
    print("🚀 COMPLETING STEP 1: KNOWLEDGE GRAPH SETUP")
    print("="*60)
    
    builder = KnowledgeGraphBuilder(driver)
    
    try:
        # 1. Create domain entities
        builder.create_domain_entities()
        
        # 2. Create semantic relationships  
        builder.create_semantic_relationships()
        
        # 3. Create document entities
        builder.create_document_entities()
        
        # 4. Create indexes for retrieval
        builder.create_indexes_for_retrieval()
        
        # 5. Implement entity linking
        builder.implement_entity_linking()
        
        # 6. Generate summary
        node_counts, rel_count = builder.create_knowledge_graph_summary()
        
        print("\n" + "="*60)
        print("🎉 STEP 1 COMPLETE!")
        print("✅ Knowledge Graph is ready for RAG system")
        print("✅ Semantic relationships established")  
        print("✅ Full-text search indexes created")
        print("✅ Entity linking implemented")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error completing Step 1: {e}")
        return False
    
    finally:
        driver.close()

if __name__ == "__main__":
    success = complete_step1()
    if success:
        print(f"\n🎯 READY FOR STEP 2: Hybrid Vector + Graph Storage")
    else:
        print(f"\n🔄 Please fix errors before proceeding to Step 2")